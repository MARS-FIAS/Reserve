#!/bin/bash


#variable used for configuration:
BASE_DIR="shell_tutorial"

echo "Warning: This script will override all content in the $BASE_DIR directroy. (if the directroy does not exist it will be created here)"
echo "please confirm by typing yes and pressing enter:"

# reads user input into the variable answer
read answer

if [ "$answer" != "yes" ] && [ "$answer" != "Yes" ]; then
	echo "ok, so nothing was changed"
	exit # so else = continiue
fi

rm -rf $BASE_DIR

mkdir $BASE_DIR

mkdir $BASE_DIR/A
mkdir $BASE_DIR/A/A1
mkdir $BASE_DIR/B

echo "No, you can use tab to autocomplete a file- or commandname. Press tab twice in order to list all possible completions." > $BASE_DIR/B/this_is_a_long_filename.do_I_seriously_need_to_type_this_ridiculously_long_filename

echo "You can use the command find $PWD/$BASE_DIR -name copy_me -type f in order to locate this file." > $BASE_DIR/A/A1/copy_me
# note that $PWD is an environment variable

# the hidden file:
echo "This file is hidden." > $BASE_DIR/.hidden_file
echo "In order to manipulate a files content one may use an editor such as nano." >> $BASE_DIR/.hidden_file
echo -e "The usage of nano is quite straightforward.\nBasically when running nano, its usage is shown at the bottom. e.g. ^X means that one have to press ctrl+x in order to close the editor." >> $BASE_DIR/.hidden_file
echo "Why you don't try it now and corrrect the spellling errors made in this file ;-)" >> $BASE_DIR/.hidden_file


echo "this is fileX." > $BASE_DIR/A/A1/fileX
echo "this file should be moved to the directroy C" >> $BASE_DIR/A/A1/fileX

echo "this is fileY." > $BASE_DIR/A/A1/fileY
echo "this file should be moved to the directroy A" >> $BASE_DIR/A/A1/fileY


#fill dir A1 with content
MAX=500
for ((i=1; i <= MAX ; i++)) ; do
	echo "this is file$i" > $BASE_DIR/A/A1/file$i
done	

for filename in $BASE_DIR/A/A1/*
do
    if [ ! -d "$filename" ]; then
        echo "this file has no special purpose." >> $filename
    fi
done

echo "this is fileZ" > $BASE_DIR/A/A1/fileZ
echo "this file has no special purpose." >> $BASE_DIR/A/A1/fileZ


# init large_file

echo "this is a large file with many lines" > $BASE_DIR/A/large_file

MAX1=2019
for ((i=2; i <= MAX1 ; i++)) ; do
	echo "this is line $i of the large file" >> $BASE_DIR/A/large_file	
done

echo "auctual useful information: files, whichs name starts with a dot \".\" are hidden. One may use -a option of ls to list them." >> $BASE_DIR/A/large_file	

MAX2=4242
for ((i=MAX1+1; i <= MAX2 ; i++)) ; do
	echo "this is line $i of the large file" >> $BASE_DIR/A/large_file	
done


#echo "additional hints:" > $BASE_DIR/hints.txt
#echo "TODO: Evaluate if additional information is required"


echo "initialization finished"
