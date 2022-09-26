#!/bin/bash
# 1: repetition
# 2: seed

echo "Cycle 1"
python3 loop_cycle1.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"

echo "Cycle 2"
python3 loop_cycle2.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"

echo "Cycle 3"
python3 loop_cycle3.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"

echo "Cycle 4"
python3 loop_cycle4.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"
cd "Cycle4"
python3 test_conv_DeepEST.py $1
cd ..
java -jar DeepEST.jar "Cycle4/rep" 1000 $1 500

echo "Cycle 5"
python3 retraining_shift_cycle_5.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"
cd "Cycle5"
python3 test_conv_DeepEST.py $1
cd ..
java -jar DeepEST.jar "Cycle5/rep" 1000 $1 500

echo "Cycle 6"
python3 retraining_shift_cycle_6.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"
cd "Cycle6"
python3 test_conv_DeepEST.py $1
cd ..
java -jar DeepEST.jar "Cycle6/rep" 1000 $1 500

echo "Cycle 7"
python3 retraining_shift_cycle_7.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"

echo "Cycle 8"
python3 loop_cycle8.py $1 $2
echo "Run the pseudo-oracle on KNIME"
read -p "Press enter to continue"