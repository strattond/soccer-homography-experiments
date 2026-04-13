Param(
  [Parameter(Mandatory = $True)] [String] $inputPath,
  [Parameter(Mandatory = $True)] [String] $labelsFolderName = 'autoLabels',
  [Parameter(Mandatory = $False)] [String] $datasetFolder = 'cvatDataset',
  [Parameter(Mandatory = $False)] [String] $extension = '.png'
)

if ( ! ( Test-Path "$labelsFolderName\labels" ) ) {
  Write-Error "Labels folder missing"
  exit 1
}
if ( Test-Path $datasetFolder ) {
  Remove-Item -Recurse -Force $datasetFolder
}
New-Item -ItemType Directory $datasetFolder
New-Item -ItemType Directory "$datasetFolder/obj_train_data"

Copy-Item "$inputPath\*$extension" "$datasetFolder/obj_train_data"
Copy-Item "$labelsFolderName\labels\*.txt" "$datasetFolder/obj_train_data"

$cocoClasses = @(
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
)

$cocoClasses | Set-Content "$datasetFolder/obj.names"

# obj.data:
$obj_data = @(
# classes = 3 # optional
  "names = obj.names",
  "train = train.txt"
)
$obj_data | Set-Content "$datasetFolder/obj.data"

$trainFile = "$datasetFolder/train.txt"
Get-ChildItem "$dataSetFolder/obj_train_data" -Filter *$extension | ForEach-Object {
  $rel = "obj_train_data/" + $_.Name
  $matchingTxt = $_.Name.Replace( $extension, '.txt' )
  if( Test-Path "$dataSetFolder/obj_train_data/$matchingTxt" ) {
    Add-Content -Path $trainFile -Value $rel
  } else {
    Write-Host "File $($_.Name) has no matching text file"
    # Remove the JPG thats missing
    Remove-Item $dataSetFolder/$rel
  }
}