"""
author: Viktor Cerny
created: 04-05-19
last edit: 24-05-19
desc: everything that has to do with training and trainingloop, but cant
be put in external lib
"""

if __name__ == '__main__':

    import torch.optim as optim
    import torch.nn as nn
    import torch
    import torchvision

    from time import time,sleep
    import os

    from lib import model
    import evaluation


    startTime = time()


    #HYPERPARAMETER / FilePaths
    weightPath = os.path.join("trainedNets","analizerCNN_ResNeXt_v0.5.0_Adam_Cross_004.pt")
    weightPathBackup = os.path.join("trainedNets", "analizerCNN_ResNeXt_v0.5.0_Adam_Cross_004_15_5.pt")
    trainDataPath = "trainDataRoot"
    testDataPath = "testDataRoot"
    batchSize = 126
    epochs = 151
    LEARNING_RATE = 0.5
    logEpoch = []


    #standard training objects creation
    AnalizerCNN = model.AnalizerCNN(8,11,2121,64,4)
    # AnalizerCNN = model.AnalizerCNN(batchSize)
    optimizer = optim.SGD(AnalizerCNN.parameters(),lr=LEARNING_RATE)
    lossFunction = nn.CrossEntropyLoss()


    #original data transformer
    transformerOrg = torchvision.transforms.ToTensor()
    #single data change transformers
    transformerBright = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter((0.1,0.8),0,0,0),
        torchvision.transforms.ToTensor()
    ])
    transformerCont = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0,(0.1,0.8),0,0),
        torchvision.transforms.ToTensor()
    ])
    transformerSat = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0,0,(0.1,0.8),0),
        torchvision.transforms.ToTensor()
    ])
    transformerHue = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0, 0, 0,(-0.5,0.5)),
        torchvision.transforms.ToTensor()
    ])
    transformerTrans = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(0,translate=(0.1,0.1)),
        torchvision.transforms.ToTensor()
    ])
    transformerCombo = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter((0.1,0.8),(0.1,0.8),(0.1,0.8),(-0.5,0.5)),
        torchvision.transforms.RandomAffine(0,translate=(0.1,0.1)),
        torchvision.transforms.ToTensor()
    ])


    #load trained Net if trained Net exists
    if os.path.isfile(weightPath):
        AnalizerCNN.load_state_dict(torch.load(weightPath))
        print("pretrained Net-weights were loaded\n")
    else:
        pass


    #gpu training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AnalizerCNN.to(device)


    #trainingLoop
    for epoch in range(1,epochs):
        #having a diffrent type of trainingSet per every settingAmount epochs
        #thus hopefully having better generalisation (overall more trainingsdata)
        settingAmount = 1

        if epoch % settingAmount == 0:
            trainingSet = torchvision.datasets.ImageFolder("trainDataRoot", transformerOrg)
            # print("training original")
        # elif epoch % settingAmount == 1:
        #     trainingSet = torchvision.datasets.ImageFolder(trainDataPath, transformerBright)
        #     print("training bright")
        # elif epoch % settingAmount == 2:
        #     trainingSet = torchvision.datasets.ImageFolder(trainDataPath, transformerCont)
        #     print("training contrast")
        # elif epoch % settingAmount == 3:
        #     trainingSet = torchvision.datasets.ImageFolder(trainDataPath, transformerSat)
        #     print("training saturation")
        # elif epoch % settingAmount == 4:
        #     trainingSet = torchvision.datasets.ImageFolder(trainDataPath, transformerHue)
        #     print("training hue")
        # if epoch % settingAmount == 0:
        #     trainingSet = torchvision.datasets.ImageFolder(trainDataPath, transformerTrans)
            # print("training translation")
        # elif epoch % settingAmount == 1:
        #     trainingSet = torchvision.datasets.ImageFolder(trainDataPath, transformerCombo)
        #     print("training Combo")

        #upgraded ram :3 now we can go 5+ workers
        trainLoader = torch.utils.data.DataLoader(trainingSet, batchSize, shuffle=True, num_workers=5,
                                                  pin_memory=True)

        #batchLoop
        for batchNum, data in enumerate(trainLoader,0):
            inputImages, classValues = data
            inputImages, classValues = inputImages.to(device), classValues.to(device)

            optimizer.zero_grad()
            output = AnalizerCNN(inputImages)
            loss = lossFunction(output,classValues)
            loss.backward()
            optimizer.step()

            # if batchNum % 100 == 0:
            #     print("Training Batch {}".format(batchNum))


        if epoch % 5 == 0 or epoch == 1:
            print("The Loss for Epoch {} is: {}".format(epoch,format(loss.item(),".6f")))
            logEpoch.append((epoch,loss.item()))



    endTime = time() - startTime
    print("We took {} seconds to train, that is {} mins.".format(format(endTime,".3f"),
                                                                 format(endTime/60,".3f")))


    #saving Weights
    try:
        torch.save(AnalizerCNN.state_dict(),weightPath)
        torch.save(AnalizerCNN.state_dict(), weightPathBackup)
        with open("logfileTraining.txt","w",encoding="utf-8") as f:
            f.write(str(logEpoch))
    except:
        print("Save was not succesfull.")
    else:
        print("Save was succesfull.")

    sleep(1)



    #evaluation
    print()
    print("Now we are evaluating.")

    evaluation.evaluation(weightPath,testDataPath)








