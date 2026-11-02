import glob, shutil
f = glob.glob('/kaggle/input/**/submission.csv', recursive=True)
print(f)
shutil.copy(f[0], '/kaggle/working/submission.csv')
