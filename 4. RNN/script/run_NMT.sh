batch = 32;

python run_NMT.py --attention false --scoring false --in_feed f --batch_size $batch
python run_NMT.py --attention global --scoring loc --in_feed f --batch_size $batch
python run_NMT.py --attention global --scoring loc --in_feed t --batch_size $batch
python run_NMT.py --attention loc_pred --scoring gen --in_feed t --batch_size $batch
python run_NMT.py --attention loc_pred --scoring gen --in_feed t --unk t --batch_size $batch