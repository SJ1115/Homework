for reverse in F T;
do
    for unk in F T;
    do
        python run_preprocess.py --reverse $reverse --unk $unk;
    done;
done;