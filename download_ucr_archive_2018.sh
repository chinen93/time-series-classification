URL=https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip
PROJECT_FOLDER=src/project
ZIP_ARCHIVE=$PROJECT_FOLDER/UCRArchive_2018.zip

curl $URL -o $ZIP_ARCHIVE
unzip $ZIP_ARCHIVE -d $PROJECT_FOLDER
