python -m venv env
source env/bin/activate

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