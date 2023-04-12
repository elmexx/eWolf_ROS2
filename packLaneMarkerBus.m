classdef packLaneMarkerBus < matlab.System & matlab.system.mixin.Propagates ...
        & matlab.system.mixin.CustomIcon
    
    properties
    % Camera sensor parameters
    Camera = struct('ImageSize',[960 1280],...
        'PrincipalPoint',[963.4416  526.8709],...
        'FocalLength',[1121.2 1131.9],...
        'Position',[1.8750 0 1.2000],...
        'PositionSim3d',[0.5700 0 1.2000],...
        'Rotation',[0 0 0],...
        'LaneDetectionRanges',[6 30],...
        'DetectionRanges',[6 50],...
        'MeasurementNoise',diag([6,1,1]));
    end
    
    properties (SetAccess='private', GetAccess='private', Hidden)
        Sensor
        LastValidLaneLeft
        LastValidLaneRight
        KalmanLeftFilter
        KalmanRightFilter
        IsLeftTrackInitialized
        IsRightTrackInitialized
        LeftPredict
        RightPredict
        % Flag to discard invalid lanes at the staritng of the simulation
        % in lane tracker
        FirstInstance   
        LaneXExtentThreshold
        LaneSegmentationSensitivity  % increased as compared to original demo
        ApproximateLaneMarkerWidth   % width specified in meters
        LaneStrengthThreshold
        MaxNumLaneMarkersToDetect
        LaneDetectionRanges
        BirdsEyeConfig
        BirdsEyeImage
        BirdsEyeBW
        VehicleROI
        FrameCount
    end
    
    methods(Access = protected)
        
        function setupImpl(obj)
            camera = obj.Camera;

        end
              
        function [lanes] = stepImpl(obj,msgs)
            
            % Pack lane boundaries to LaneSensor bus
            lanes = packLaneBoundaryDetections(obj, msgs);

        end

        %------------------------------------------------------------------
        % packLaneBoundaryDetections method packs Pack detections into
        % format expected by LaneFollowingDecisionLogicandControl. 
        function lanes  = packLaneBoundaryDetections(obj, msgs)

            leftEgoParameters = zeros(1,3); % y=Ax^2+Bx+C
            rightEgoParameters = zeros(1,3);
            leftEgoParameters(1) = typecast( uint8(msgs(1:8)) , 'double');
            leftEgoParameters(2) = typecast( uint8(msgs(9:16)) , 'double');
            leftEgoParameters(3) = typecast( uint8(msgs(17:24)) , 'double');
            rightEgoParameters(1) = typecast( uint8(msgs(25:32)) , 'double');
            rightEgoParameters(2) = typecast( uint8(msgs(33:40)) , 'double');
            rightEgoParameters(3) = typecast( uint8(msgs(41:48)) , 'double');
            
            
            leftEgoBoundary = parabolicLaneBoundary(leftEgoParameters);
            rightEgoBoundary = parabolicLaneBoundary(rightEgoParameters);
            % Preallocate struct expected by controller
            DefaultLanes = struct('Curvature',{single(0)},...
                'CurvatureDerivative',{single(0)},...
                'HeadingAngle',{single(0)},...
                'LateralOffset',{single(0)},...
                'Strength',{single(0)},...
                'XExtent',{single([0,0])},...
                'BoundaryType',{LaneBoundaryType.Unmarked});

            field1 = 'Left'; field2 = 'Right';
            lanes = struct(field1,DefaultLanes,field2,DefaultLanes);           
            zeroStrengthLane = DefaultLanes;

            % Pack detections into struct
            lanes.Left  = packLaneBoundaryDetection(leftEgoBoundary);
            lanes.Right = packLaneBoundaryDetection(rightEgoBoundary);

            % Shift detections to vehicle center as required by controller
            % Note: camera.PositionSim3d(1) represents the X mount location of the
            %       camera sensor with respect to the vehicle center
            if nnz(leftEgoBoundary.Parameters)
                lanes.Left.LateralOffset(:) = polyval(...
                    leftEgoBoundary.Parameters, -obj.Camera.PositionSim3d(1));
                % Lane to left should always have positive lateral offset
                if lanes.Left.LateralOffset < 0
                    lanes.Left = zeroStrengthLane;
                end
            end
            if nnz(rightEgoBoundary.Parameters)
                lanes.Right.LateralOffset(:) = polyval(...
                    rightEgoBoundary.Parameters, -obj.Camera.PositionSim3d(1));
                % Lane to right should always have negative lateral offset
                if lanes.Right.LateralOffset > 0
                    lanes.Right = zeroStrengthLane;
                end
            end
        end

        function [lanes] = getOutputSizeImpl(obj) %#ok<MANU>
            % Return size for each output port
            lanes = 1;
        end
        
        function [lanes] = getOutputDataTypeImpl(obj) %#ok<MANU>
            % Return data type for each output port
            lanes = "LaneSensor";
        end
        
        function [lanes] = isOutputComplexImpl(obj) %#ok<MANU>
            % Return true for each output port with complex data
            lanes= false;
        end
        
        function [lanes] = isOutputFixedSizeImpl(obj) %#ok<MANU>
            % Return true for each output port with fixed size
            lanes = true;
        end
    end
    
    methods(Access = protected, Static)
        function header = getHeaderImpl
            % Define header panel for System block dialog
            header = matlab.system.display.Header(....
                "Title","Pack Lane Bus",...
                "Text",...
                "Pack lanes parameters from Jetson to Simulink Lane Bus." + newline + newline);
        end

        function flag = showSimulateUsingImpl
            % Return false if simulation mode hidden in System block dialog
            flag = true;
        end
    end
end

function detection = packLaneBoundaryDetection(boundary)
% Parameters of parabolicLaneBoundary object = [A B C]
%  corresponds to the three coefficients of a second-degree
%  polynomial equation:
%                y = Ax^2 + Bx + C
% Comparing this equation with lane model using 2nd order
% polynomial approximation:
%  y = (curvature/2)*(x^2) + (headingAngle)*x + lateralOffset
%
% This leads to the following relationship
%   curvature           = 2 * A = 2 * Parameters(1)  (unit: 1/m)
%   headingAngle        = B     = Parameters(2)      (unit: radians)
%   lateralOffset       = C     = Parameters(3)      (unit: meters)
%

    % Default lane of zero strength
    detection = struct('Curvature',{single(0)},'CurvatureDerivative',...
        {single(0)},'HeadingAngle',{single(0)},'LateralOffset',{single(0)},...
        'Strength',{single(0)},'XExtent',{single([0,0])},...
        'BoundaryType',{LaneBoundaryType.Unmarked});
    if nnz(boundary.Parameters)
        detection.Curvature(:)     = 2 * boundary.Parameters(1);
        detection.HeadingAngle(:)  = boundary.Parameters(2); % Coordinate transform
        detection.LateralOffset(:) = boundary.Parameters(3); % Coordinate transform
        detection.Strength(:)      = boundary.Strength;
        detection.XExtent(:)       = boundary.XExtent;
        detection.BoundaryType(:)  = boundary.BoundaryType;
    end
end