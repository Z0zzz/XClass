set -e

gpu=$1
dataset=$2
for m in 30 29 27 23 15
do
    CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset} --mask ${m}
    CUDA_VISIBLE_DEVICES=${gpu} python class_oriented_document_representations.py --dataset_name ${dataset} --mask ${m}
    python document_class_alignment.py --dataset_name ${dataset} --mask ${m}
    python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100-masked-${m}
    python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42-masked-${m}
    python prepare_text_classifer_training.py --dataset_name ${dataset} --mask ${m} 
    ./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.bbu-12.mixture-100.42.0.5-masked-${m}
    python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100-masked-${m}
    python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42-masked-${m}
done
