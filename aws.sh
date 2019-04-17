upload(){
    aws s3 cp $1 s3://oscar-multi-doc-summarizer-thesis/$1
}

download(){
    aws s3 cp s3://oscar-multi-doc-summarizer-thesis/$1 $1
}    


aws_ls(){
    aws s3 ls s3://oscar-multi-doc-summarizer-thesis/
}

structure_experiment(){
    mv raw_train.json raw_train_$1.json
    mv raw_val.json raw_val_$1.json
    mv train_loss.pickle train_loss_$1.pickle
    mv val_loss.pickle val_loss_$1.pickle
    mv weights_latest weights_$1
    upload train_loss_$1.pickle
    upload val_loss_$1.pickle
    mv raw_train_$1.json raw
    mv raw_val_$1.json raw
    mv train_loss_$1.pickle pickles
    mv val_loss_$1.pickle pickles
    mv weights_$1 weights
}
