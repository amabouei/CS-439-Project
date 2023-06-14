P="logs2"
list="cifar10 cifar100 mnist"
epochs=1
iswandb=0

###SGD

for i in $list; do 
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.1 --opt SGD --dataset $i --epochs $epochs
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.05 --opt SGD --dataset $i --epochs $epochs
done




#####ADAM

for i in $list; do 
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.01 --opt ADAM --dataset $i --epochs $epochs
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.005 --opt ADAM --dataset $i --epochs $epochs
done



####adahessian

for i in $list; do 
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.15 --opt adahessian --dataset $i --epochs $epochs
done




###sophia

for i in $list; do 
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.01  --opt Sophia --dataset $i  --epochs $epochs
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.005 --opt Sophia --dataset $i  --epochs $epochs
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.001 --opt Sophia --dataset $i  --epochs $epochs
	python3 train_classifier.py --path $P --wandb $iswandb --lr 0.0001 --opt Sophia --dataset $i  --epochs $epochs
done	
