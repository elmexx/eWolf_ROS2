#!/bin/bash

URL_FILE="check_links.txt"
DOWNLOAD_DIR="./download"
REMAINING_FILE="remaining_urls.txt"
MAX_USAGE_GB=100 

mkdir -p "$DOWNLOAD_DIR"

counter=0
stop=false

cp "$URL_FILE" "$REMAINING_FILE"
TMP_FILE="tmp_urls.txt"
cp "$REMAINING_FILE" "$TMP_FILE"

while IFS= read -r url
do
    counter=$((counter+1))
    url=$(echo "$url" | tr -d '\r' | xargs)
    filename=$(echo "$url" | grep -o 'FN-ZF[^?]*')
    echo "Downloading $counter: $filename"
    wget -O "$filename" "$url"

    used_gb=$(du -sBG "$DOWNLOAD_DIR" | awk '{print $1}' | tr -dc '0-9')
    echo "Used Space: ${used_gb}GB"

    if [ "$used_gb" -ge "$MAX_USAGE_GB" ]; then
        echo "Not enough space, stop download"
        echo "This is $counter"
        stop=true
        break
    fi

    sed -i '1d' "$REMAINING_FILE"

done < "$TMP_FILE"

if [ "$stop" = true ]; then
    remaining=$(wc -l < "$REMAINING_FILE")
    echo "Downloaded to $counter"
    echo "Remain $remaining links not downloaded, stored in $REMAINING_FILE"
else
    echo "Download complete with $counter links"
    rm -f "$REMAINING_FILE"
fi

rm -f "$TMP_FILE"

