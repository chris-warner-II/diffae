# # Make sure you are in basic conda environment with only python and pip
# conda activate basic_py3.9

# #Run these on Catscan
# ln -s ~/Data/ddpm store
# ln -s ~/Data/ddpm/datasets/diffae/datasets/ datasets
# ln -s ~/Data/ddpm/models/diffae/checkpoints/ checkpoints
# ln -s ~/.cache ~/Data/.cache

# #Run these on Laptop
# ln -s ../store store
# ln -s ../store/datasets/diffae/datasets/ datasets
# ln -s ../store/models/diffae/checkpoints/ checkpoints


python -m venv store/venv/diffae
source store/venv/diffae/bin/activate

pip install --upgrade pip

pip install -r requirements.txt


# Convert .ipynb to .py to run them from commandline
conv_files=("sample" "manipulate" "interpolate" "autoencoding")
for element in "${conv_files[@]}"; do
	if [ ! -f "$element.py" ]; then
		echo "Converting $element from .ipynb to .py"
		jupyter nbconvert --to python $element.ipynb
	else
		echo "$element.py already exists"
	fi
done
