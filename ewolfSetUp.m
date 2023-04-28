% Set up Script for the eWolf
%
% This script loads necessary control
% constants and sets up the buses required for the referenced model


%% General Model Parameters
% Ts = 0.1;               % Simulation sample time                (s)

%% Ego Car Parameters
% Dynamics modeling parameters
m       = 1666;         % Total mass of vehicle                 (kg) laut Fzg.schein 1750kg Leergewicht
Iz      = 2641;         % Yaw moment of inertia of vehicle Iz=0.1269mRL (Kgm^2)
lf      = 1.2;          % Longitudinal distance from Center of Gravity to front tires (m)
lr      = 1.6;          % Longitudinal distance from Center of Gravity to rear tires  (m)
Cf      = 19000;        % Cornering stiffness of front tires    (N/rad)
Cr      = 33000;        % Cornering stiffness of rear tires     (N/rad)
tau     = 0.5;          % Time constant for longitudinal dynamics 1/s/(tau*s+1)
default_spacing = 10;   % Default spacing                       (m)
max_ac  = 2;            % Maximum acceleration                  (m/s^2)
min_ac  = -3;           % Minimum acceleration                  (m/s^2)
max_steer = 0.26;       % Maximum steering                      (rad)
min_steer = -0.26;      % Minimum steering                      (rad) 
max_dc  = -10;          % Maximum deceleration                  (m/s^2)
tau2    = 0.07;         % Longitudinal time constant (brake)    (N/A)

%% Ego struct
ego.FrontOverhang = 0.9;% Ego front overhang                    (m)
ego.RearOverhang = 1;   % Ego rear overhang                     (m)
ego.Length = 4.7;       % Ego length                            (m)
ego.Position = [0,0,0]; % Ego initial position                  (m)
ego.Velocity = [0.1,0,0]; % Ego initial velocity                  (m)
ego.Roll = 0;           % Ego initial roll                      (rad)
ego.Pitch = 0;          % Ego initial pitch                     (rad)
ego.Yaw = 0;            % Ego initial yaw                       (rad)
egoVehDyn = egoVehicleDynamicsParams(ego); % Ego
assignin('base','egoVehDyn', egoVehDyn);

x0_ego = ego.Position(1);
y0_ego = ego.Position(2);
v0_ego = ego.Velocity(1);
yaw0_ego = ego.Yaw;

%% Sensor parameters
assignin('base','camera',cameraParams(ego));

%% Controller parameter
PredictionHorizon = 30; % Number of steps for preview    (N/A)

%% Bus Creation
% Create buses for lane sensor and lane sensor boundaries
createLaneSensorBuses;
createBusVehiclePose;
% Create the bus of actors from the scenario reader
% modelName = 'D_20210914_Modell_Inbetriebnahme_BreakOutBox_Jetson';
% wasModelLoaded = bdIsLoaded(modelName);
% if ~wasModelLoaded
%     load_system(modelName)
% end


%% Create scenario and road specifications
% [allData, scenario, sensors] = Campus_Simulation_Test();
% loaddata = load('campus_scenario.mat');
% scenario = loaddata.scenario;

% %% Simulink Variant System
% defaultVisionVariant   = "Vision_Simulink";
% validVisionVariants = [...
%     "Vision_Jetson";...
%     "Vision_Simulink"];
% VisionVariant = defaultVisionVariant;
% assignin('base','visionVariant',VisionVariant);

%% Function

function egoVehDyn = egoVehicleDynamicsParams(ego)
%egoVehicleDynamicsParams vehicle dynamics parameters from scenario
%
% Scenario is in ISO 8855 (North-West-Up) with respect to rear axle
% Returns struct in SAE J670E (North-East-Down) with respect to
% center of gravity (vehicle center)
%
%  egoVehDyn.X0            % Initial position X (m)
%  egoVehDyn.Y0            % Initial position Y (m)
%  egoVehDyn.Yaw0          % Initial yaw (rad)
%  egoVehDyn.VLong0        % Initial longitudinal velocity(m/sec)
%  egoVehDyn.CGToFrontAxle % Distance center of gravity to front axle (m)
%  egoVehDyn.CGToRearAxle  % Distance center of gravity to rear axle (m)

% Ego in ISO 8855 (North-West-Up) with respect to rear axle
% ego = scenario.Actors(1);

% Shift reference position to center of gravity (vehicle center)
position_CG = driving.scenario.internal.Utilities.translateVehiclePosition(...
    ego.Position,...     % Position with respect to rear axle (m)
    ego.RearOverhang,... % (m)
    ego.Length,...       % (m)
    ego.Roll,...         % (deg)
    ego.Pitch,...        % (deg)
    ego.Yaw);            % (deg)

% Translate to SAE J670E (North-East-Down)
% Adjust sign of y position to 
egoVehDyn.X0  =  position_CG(1); % (m)
egoVehDyn.Y0  = -position_CG(2); % (m)
egoVehDyn.VX0 =  ego.Velocity(1); % (m)
egoVehDyn.VY0 = -ego.Velocity(2); % (m)

% Adjust sign and unit of yaw
egoVehDyn.Yaw0 = -deg2rad(ego.Yaw); % (rad)

% Longitudinal velocity 
egoVehDyn.VLong0 = hypot(egoVehDyn.VX0,egoVehDyn.VY0); % (m/sec)

% Distance from center of gravity to axles
egoVehDyn.CGToFrontAxle = ego.Length/2 - ego.FrontOverhang;
egoVehDyn.CGToRearAxle  = ego.Length/2 - ego.RearOverhang;

end

function camera = cameraParams(ego)
% Camera sensor parameters

%     Camera = struct('ImageSize',[960 1280],...
%         'PrincipalPoint',[963.4416  526.8709],...
%         'FocalLength',[1121.2 1131.9],...
%         'Position',[1.8750 0 1.2000],...
%         'PositionSim3d',[0.5700 0 1.2000],...
%         'Rotation',[0 0 0],...
%         'LaneDetectionRanges',[6 30],...
%         'DetectionRanges',[6 50],...
%         'MeasurementNoise',diag([6,1,1]));

camera.NumColumns      = 640;     % Number of columns in camera image - width
camera.NumRows         = 480;     % Number of rows in camera image - height
camera.FieldOfView     = [100,45]; % Field of view (degrees)
camera.ImageSize       = [camera.NumRows, camera.NumColumns];
camera.PrincipalPoint  = [320  240];
camera.FocalLength     = [322.28 322.28];
camera.Position        = ...      % Position with respect to rear axle (m)
    [ 1.1750, ...                 %  - X (by the rear-view mirror)
      0,...                       %  - Y (center of vehicle width)
      1.1];                       %  - Height
camera.PositionSim3d   = ...      % Position with respect to vehicle center (m)
    camera.Position - ...         %  - Reduce position X by distance from vehicle center to rear axle
    [ ego.Length/2 - ego.RearOverhang,...
      0, 0];
camera.Rotation = [0, -5, 0];          % Rotation [roll, pitch, yaw] (deg)
camera.DetectionRanges  = [3 50];     % Full range of camera (m)
camera.LaneDetectionRanges  = [2 25]; % Range to detect lane markings (m)
camera.MeasurementNoise = diag([...   % Measurement noise for vehicle detection
    6,...                             % x is noisier than y
    1,...                             % y is most accurate
	1]); ...                          % z is also accurate
camera.MinObjectImageSize = [10,10];  % Min object size for probabilistic sensor (pixels)
end