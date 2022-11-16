
# Emilie's Weekly Meeting Notes

* [13 October 2022](#date-30-october-2022)
* [20 October 2022](#date-20-october-2022)
* [27 October 2022](#date-27-october-2022)
* [03 November 2022](#date-03-november-2022)
* [09 November 2022](#date-09-november-2022)

### Date: 13 October 2022

#### Who did you help this week?

N/A

#### Who helped you this week?

* Veronika helped guiding me through how the project can look like and what component should be involved.
* She also guided me to where and what data i could and should use. 


#### What did you achieve?

* I read the papers suggested in the project description link: https://dasya.itu.dk/for-students/proposals/current/pretraining_strategies/ 
* I started gathering data needed. 

#### What did you struggle with?

* I struggled with the infrastructure of the autoencoder, such as how to build it to achieve the best results possible. I am also struggling with running time on my local laptop. 

#### What would you like to work on next week?

* I would like to keep improving my autoencoder on the cifar100 dataset, possible try using my infrastructure on the entire Imagenet dataset. 

#### Where do you need help from Veronika & Dovile?

* How to improve both my autoencoder and finetuning
* Additionally suggestions for medical images.

### Date: 20 October 2022

#### Who did you help this week?

N/A

#### Who helped you this week?

* Dovile helped me this week I have had a lot of problems with the running time, when training my autoencoder so I needed to move the process over to the hpc.

#### What did you achieve?

* We thought we achieved to solve the enviroment for the hpc to have the autoencoder run on the gpu's.

#### What did you struggle with?

* I struggled a lot with the hpc and getting my code to run on gpu as there was some fault in the virtual envoriment, and the way tensorflow was installed. Whenever Dovile and I thought we had solved it, the enviroment would collapse and we would have to start debugging the whole problem again. 

#### What would you like to work on next week?

* Getting the gpu to work on the hpc. 

#### Where do you need help from Dovile & Veronika?

* Getting the gpu to work on the hpc. 

### Date: 27 October 2022

#### Who did you help this week?

N/A

#### Who helped you this week?

* No one we had no meeting this week. 

#### What did you achieve?

* THE GPU FINALLLY WORKED and i was back on track to improve my autoencoder and fine tuning on the x-ray dataset. 
* Additionally I did a test run on the cifar100 dataset and achieved a test score, on the chest x-ray, of 95%. 


#### What did you struggle with?

* Stategies for reading Imagenet, reaching convergence on imagenet as it kept stagnatig. 

#### What would you like to work on next week?

* How to make the autoecoder converge on imagenet.

#### Where do you need help from Veronika a& Dovile?

* suggestions as to why the autoencoder is not converging on imagenet. 

### Date: 03 November 2022

#### Who did you help this week?

N/A

#### What did you achieve?

* I finally mangaged to improve my autoencoder on imagenet, and it is now running on the GPU!


#### What did you struggle with?

* Figuring out what would help boost the loss on the autoencoder. 

#### What would you like to work on next week?

* I am doing finetuning on all 3 medical datasets I am using. And i am gonna start the training for the autoencoder trained on medical images when they are avaliable. 

#### Where do you need help from Veronika & Dovile?

* I am not entirely sure. 

### Date: 09 November 2022

#### Who did you help this week?

N/A

#### What did you achieve?

* I am still dealing with the the same as last week. 


#### What did you struggle with?

* Simple bugs and mistakes in the code.

#### What would you like to work on next week?

* Training for the autoencoder trained on medical images when they are avaliable. 

#### Where do you need help from Veronika & Dovile?

* I am not entirely sure. 

### Date: 17 November 2022

#### Who did you help this week?

N/A

#### What did you achieve?

* I achived to make a final fine tuning on the chest x-ray with 5-fold cross validation, using the imagenet autoencoder. this also means that i have a done architecture for the radimagenet autoeencoder which should speed up the rest of the process.   


#### What did you struggle with?

* I struggled with the cross validation as the way i read in my images made it diffult to implement my usual sklearn crossvalidation go to. I found a work around and made it work. There also seem to be a lot of traffic on the gpu, so running eksperiments take ekstra time this week. 

#### What would you like to work on next week?

* finish all finetuning experiments and have the finished radimagenet autoencoder uploaded as well. 

#### Where do you need help from Veronika & Dovile?

* Feedback on how far I am in the process. I sometimes lack the ability to see if I am on time or not. I am estimating that I need at least 14 days to just write the paper. Meaning I have the rest of November that finish the code? 


