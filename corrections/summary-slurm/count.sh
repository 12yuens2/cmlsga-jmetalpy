for d in /ssdfs/users/sy6u19/data-100000evals-30runs-corrections-normals/cMLSGA/*/; do echo "$(echo "$d" | awk -F'[/ ]+' '{print $F}') $(ls -1 "$d" | wc -l)"; done
