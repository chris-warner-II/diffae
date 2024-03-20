python -m venv env
source env/bin/activate

pip install --upgrade pip

pip install -r requirements.txt


# Create symlink to store directory
storage_dir="../store" # on my laptop
storage_dir="~/Data/ddpm" # on CatScan
if [ ! -d "store" ]; then
	echo "creating symlink from store to $storage_dir"
	ln -s $storage_dir store
fi


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
