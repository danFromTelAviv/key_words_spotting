# key words spotting
pytorch implementation of :
Coucke et al., 2018, "Efficient keyword spotting using dilated convolutions
and gating"
http://150.162.46.34:8080/icassp2019/ICASSP2019/pdfs/0006351.pdf

* we did not use their trick with labeling part of the positive samples as positive using a VAD. Instead we used maxpooling. We found that using some changes to the proposed architecture we can still get similar results without finetuning a bunch of hyperparameters and using a VAD. 


Utilizing the data found at : 
https://github.com/snipsco/keyword-spotting-research-datasets


## Instructions for how to use the code.

The repo is very minimal. 
* make sure all of the packages needed are installed (refer to requirements.txt)
* download the snips keywords dataset 
* change the path in main.py to the path of the snip dataset.
* run main.py

as is:
```buildoutcfg
epoch: 17, loss: 0.021094020111433718, val_loss:0.019179460771668412, acc: 0.9086629001883191, val_acc: 0.9324358974358976
```

with some architecture changes that I can not share ; )
```buildoutcfg
epoch: 19, loss: 0.0041030201199985334, val_loss:0.0021393651105013197, acc: 0.9853107344632844, val_acc: 0.9932051282051269
```




