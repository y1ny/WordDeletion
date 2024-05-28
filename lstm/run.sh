
for s in {1,2,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800};
do
    for n in {0..9};
    do
        python construct_deletion.py -n $s -i $n -o 'data'
        python run.py -m preprocess -n $s
        python run.py -m train -i $n -n $s
        python run.py -m test -i $n -n $s
    done
done