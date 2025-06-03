#!/bin/bash

set -e  # Exit on error

start_stage=1
stop_stage=8

echo "Running pipeline from stage $start_stage to $stop_stage"

# Stage 1: Data Preparation
if [ $start_stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "===== Stage 1: Preparing Data ====="
  if [ ! -f "data/en_files/source" ] || [ ! -f "data/hi_files/target" ]; then
    mkdir -p data/en_files data/hi_files
    python3 prepare_data.py
    echo "Stage 1 complete: Data prepared."
  else
    echo "Stage 1 skipped: Data already exists."
  fi
fi

# Stage 2: Filtering
if [ $start_stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "===== Stage 2: Filtering ====="
  if [ ! -f "source-filtered.en" ] || [ ! -f "target-filtered.hi" ]; then
    python3 MT-Preparation/filtering/filter.py data/en_files/source data/hi_files/target en hi
    echo "Stage 2 complete: Filtering done."
  else
    echo "Stage 2 skipped: Filtered files already exist."
  fi
fi

# Stage 3: Subword Model Training
if [ $start_stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "===== Stage 3: Training Subword Models ====="
  if [ ! -f "source.model" ] || [ ! -f "target.model" ]; then
    python3 MT-Preparation/subwording/1-train_unigram.py data/en_files/source-filtered.en data/hi_files/target-filtered.hi
    echo "Stage 3 complete: Subword models trained."
  else
    echo "Stage 3 skipped: Subword models already exist."
  fi
fi

# Stage 4: Apply Subwording
if [ $start_stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "===== Stage 4: Applying Subwording ====="
  python3 MT-Preparation/subwording/2-subword.py source.model target.model data/en_files/source-filtered.en data/hi_files/target-filtered.hi
  echo "Stage 4 complete: Subwording applied."
fi

# Stage 5: Splitting
if [ $start_stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "===== Stage 5: Train-Dev-Test Split ====="
  python3 MT-Preparation/train_dev_split/train_dev_test_split.py 2000 2000 data/en_files/source-filtered.en.subword data/hi_files/target-filtered.hi.subword
  echo "Stage 5 complete: Splitting done."
fi

# Stage 6: Build Vocabulary
if [ $start_stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "===== Stage 6: Building Vocabulary ====="
  onmt_build_vocab -config config/config.yaml -n_sample -1 -num_threads 2
  echo "Stage 6 complete: Vocabulary built."
fi

# Stage 7: Training NMT Model
if [ $start_stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "===== Stage 7: Training NMT Model ====="
  onmt_train -config config/config.yaml -tensorboard -tensorboard_log_dir runs
  echo "Stage 7 complete: Model trained."
fi

# Stage 8: Translation and Evaluation
if [ $start_stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "===== Stage 8: Translation and Evaluation ====="
  onmt_translate -model models/model.enhi_step_5000.pt \
                 -src data/en_files/source-filtered.en.subword.test \
                 -output data/hi_files/en.hi.translated \
                 -gpu 0 -min_length 1

  python3 MT-Preparation/subwording/3-desubword.py target.model data/hi_files/en.hi.translated
  python3 MT-Preparation/subwording/3-desubword.py target.model data/hi_files/target-filtered.hi.subword.test
  python3 compute-results.py data/hi_files/target-filtered.hi.subword.test.desubword data/hi_files/en.hi.translated.desubword

  echo "Stage 8 complete: Evaluation done."
fi

echo "Pipeline completed from stage $start_stage to $stop_stage."
