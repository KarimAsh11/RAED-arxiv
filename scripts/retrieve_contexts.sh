#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/bin/activate entity-rag

ENCODER=riccorl/aida-e5-base-v2-index-19-july-self-wikipedia
INDEX=riccorl/aida-e5-base-v2-index-19-july-self-wikipedia-index

DATA_FOLDER=data/entity-rag/data
OOD_FOLDER=$DATA_FOLDER/ood
OUTPUT_FOLDER=data/entity-rag/data/aida-e5-base-v2-index-19-july-self-wikipedia/aida+ood+zelda_contexts_en.similarity.goldenindex.06-july.top-10
 
export GOLDENRETRIEVER_CACHE_DIR=/media/data/golden_retriever_cache
export HF_HOME=/media/data/hf_cache 

echo "Retrieving aida-train-kilt-Wikipedia.DM.v4.jsonl"
python entity_rag/retriever/retrieve_aida.py \
    --encoder $ENCODER \
    --index $INDEX \
    --input_path $DATA_FOLDER/aida-train-kilt-Wikipedia.DM.v4.jsonl \
    --output_path $OUTPUT_FOLDER/aida-train-kilt-Wikipedia.DM.v4.retrieved.jsonl \
    --index_device cpu \
    --precision 16 \
    --index_precision 32

echo "Retrieving aida-dev-kilt-Wikipedia.DM.v4.jsonl"
python entity_rag/retriever/retrieve_aida.py \
    --encoder $ENCODER \
    --index $INDEX \
    --input_path $DATA_FOLDER/aida-dev-kilt-Wikipedia.DM.v4.jsonl \
    --output_path $OUTPUT_FOLDER/aida-dev-kilt-Wikipedia.DM.v4.retrieved.jsonl \
    --index_device cpu \
    --precision 16 \
    --index_precision 32

echo "Retrieving aida-test-kilt-Wikipedia.DM.v4.jsonl"
python entity_rag/retriever/retrieve_aida.py \
    --encoder $ENCODER \
    --index $INDEX \
    --input_path $DATA_FOLDER/aida-test-kilt-Wikipedia.DM.v4.jsonl \
    --output_path $OUTPUT_FOLDER/aida-test-kilt-Wikipedia.DM.v4.retrieved.jsonl \
    --index_device cpu \
    --precision 16 \
    --index_precision 32

iterate over jsonl files in the folder
for FILE in $OOD_FOLDER/*.jsonl; do
    echo "Retrieving $FILE"
    python entity_rag/retriever/retrieve_aida.py \
        --encoder $ENCODER \
        --index $INDEX \
        --input_path $FILE \
        --output_path $OUTPUT_FOLDER/$(basename $FILE).retrieved.jsonl \
        --index_device cpu \
        --precision 16 \
        --index_precision 32
        
done