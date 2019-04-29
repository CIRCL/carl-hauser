# Examples

## Few flags equivalent to all flags
pipenv run screen -d  -S lib_testing -m python3 ./launcher.py -v -i ../../INPUT/raw_phishing/ -o /mnt/ssd/OUTPUT/ -gt ../../INPUT/Phishing_vol_1.json -t PNG -sp -ao -aa

## All flags
pipenv run screen -d  -S lib_testing -m python3 ./launcher.py -v -i ../../INPUT/raw_phishing/ -o /mnt/ssd/OUTPUT/ -gt ../../INPUT/Phishing_vol_1.json -t PNG -tldr -tldr_latex -p -tldr_pair -im -pm -sp -Ov -ih -tlsh -orb -ob -void
