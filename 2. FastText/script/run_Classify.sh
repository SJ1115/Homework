#ag sogou dbpedia yelp_p yelp_f yahoo amazon_p amazon_f
for data in ag sogou dbpedia yelp_p yelp_f yahoo amazon_p amazon_f;
do
    for bool in T ;#F;
    do
        python run_Classify.py --data $data --bigram $bool;
    done;
done;