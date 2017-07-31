#!/usr/bin/bash

# written by Justin Su

scorer=./reference-coreference-scorers-8.01/scorer.pl
dev_path=./conll-2012/dev/
test_path=./conll-2012/test/
dev_gold_suffix=auto_conll
test_gold_suffix=gold_conll
dev_output_suffix=auto_conll_out
test_output_suffix=gold_conll_out
dev_results=./dev_results.txt
test_results=./test_results.txt
metric=muc

eval () {
	for path in $(find $1 -type d -links 2)
	do 
		for gold in $path/*$2
			do 
				output=${gold/$2/$3}
				if [ -e $output ]
				then
					$scorer $4 $gold $output >> ./tmp.txt
				fi
			done
	done

	cat ./tmp.txt | egrep -i "\b(Coreference: )" >> $5

	rm ./tmp.txt
}


eval $dev_path $dev_gold_suffix $dev_output_suffix $metric $dev_results
eval $test_path $test_gold_suffix $test_output_suffix $metric $test_results

echo "DEV SET PERFORMANCE"
python results_parser.py -i $dev_results
rm $dev_results

echo ""
echo "TEST SET PERFORMANCE"
python results_parser.py -i $test_results
rm $test_results
