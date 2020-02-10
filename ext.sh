for f in $1/*; 
    do 
        org_filename=$(basename -- "$f");
        extension="${org_filename##*.}";
        filename="${org_filename%.*}";
        if [ $extension == 'jfif' ]
        then
            mv "$1/$org_filename" "$1/$filename.jpg"
        elif [ $extension == "gif" ]
        then
             mv "$1/$org_filename" "$1/$filename.jpg"
        fi
done
