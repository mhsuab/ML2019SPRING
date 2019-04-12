mkdir models
cd models
wget https://github.com/mhsuab/ML2019SPRING/releases/download/models/try10.h5
wget https://github.com/mhsuab/ML2019SPRING/releases/download/models/try11.h5
wget https://github.com/mhsuab/ML2019SPRING/releases/download/models/try12.h5
wget https://github.com/mhsuab/ML2019SPRING/releases/download/models/try13.h5
wget https://github.com/mhsuab/ML2019SPRING/releases/download/models/try15.h5
cd ..
python test.py $1 $2
