Param(
  [Parameter(Mandatory = $True)] [String] $inputPath,
  [Parameter(Mandatory = $False)] [String] $model = 'models/yolo26x.pt',
  [Parameter(Mandatory = $False)] [String] $outputFolderName = 'autoLabels',
  [Parameter(Mandatory = $False)] [String] $extension = '.png'
)

& yolo predict model="$model" source="$inputPath/*$extension" project="$outputFolderName" save=True save_txt=True save_conf=False classes="0,32"
