#!/usr/bin/env bash

# Read data_text_dir path from a config file.
CURDIR=$(cd $(dirname $0); pwd)
source <(grep TEXTDIR ${CURDIR}/../config.ini | sed 's/ *= */=/g')

# Text preprocessing.
# 1. Remove blank lines.
# 2. Remove <doc id ... line and its next line (title of an article).
# 3. Replace </doc> line with a blank line.
# 4. Break line at each 。 (but not 。」 or 。）...) position.
# 5. Remove lines with the head 。(these lines are not meaningful).
for FILE in $( find ${TEXTDIR} -name "wiki_*" ); do
    echo "Processing ${FILE}"
    sed -i -e '/^$/d; /<doc id/,+1d; s/<\/doc>//g; s/。\([^」|)|）|"]\)/。\n\1/g' ${FILE}
    sed -i -e '/^。/d' ${FILE}
done

# Concat all text files in each text directory.
for DIR in $( find ${TEXTDIR} -mindepth 1 -type d ); do
    echo "Processing ${DIR}"
    for f in $( find ${DIR} -name "wiki_*" ); do cat $f >> ${DIR}/all.txt; done 
done