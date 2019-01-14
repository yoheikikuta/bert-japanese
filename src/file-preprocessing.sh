#!/usr/bin/env bash

# Read data_text_dir path from a config file.
CURDIR=$(cd $(dirname $0); pwd)
source <(sed -n '/^\[DATA\]/,/^\[/p' ${CURDIR}/../config.ini | grep TEXTDIR | sed 's/ *= */=/g')

# Text preprocessing.
# 1-1. Remove blank lines.
# 1-2. Remove <doc id ... line and its next line (title of an article).
# 1-3. Replace </doc> line with a blank line.
# 2-1. Remove spaces at the end of each line.
# 2-2. Break line at each 。, but not at 。」 or 。）, position.
# 2-3. Remove spaces at the head of each line.
# 3. Remove lines with the head 。(these lines are not meaningful).
# 4. Convert upper case characters to lower case ones.
for FILE in $( find ${TEXTDIR} -name "wiki_*" ); do
    echo "Processing ${FILE}"
    sed -i -e '/^$/d; /<doc id/,+1d; s/<\/doc>//g' ${FILE}
    sed -i -e 's/ *$//g; s/。\([^」|)|）|"]\)/。\n\1/g; s/^[ ]*//g' ${FILE}
    sed -i -e '/^。/d' ${FILE}
    sed -i -e 's/\(.*\)/\L\1/' ${FILE}
done

# Concat all text files in each text directory.
for DIR in $( find ${TEXTDIR} -mindepth 1 -type d ); do
    echo "Processing ${DIR}"
    for f in $( find ${DIR} -name "wiki_*" ); do cat $f >> ${DIR}/all.txt; done 
done
