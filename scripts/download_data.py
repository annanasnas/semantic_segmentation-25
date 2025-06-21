cityscapes_id = "1OlElYRhKovWEc8gu32E8at-xNfEQniGq"

!pip install gdown
!gdown --id {cityscapes_id} -O "Cityscapes.zip"
!unzip -q "/content/Cityscapes.zip" -d "/content"