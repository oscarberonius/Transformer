upload(){
    aws s3 cp $1 s3://oscar-multi-doc-summarizer-thesis/$1
}

download(){
    aws s3 cp s3://oscar-multi-doc-summarizer-thesis/$1 $1
}    


aws_ls(){
    aws s3 ls s3://oscar-multi-doc-summarizer-thesis/
}
