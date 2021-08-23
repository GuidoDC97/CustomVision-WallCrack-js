// Import modules
const util = require('util');
const fs = require('fs');
const TrainingApi = require("@azure/cognitiveservices-customvision-training");
const PredictionApi = require("@azure/cognitiveservices-customvision-prediction");
const msRest = require("@azure/ms-rest-js");
const {BlobServiceClient} = require('@azure/storage-blob');
const ConfusionMatrix = require('ml-confusion-matrix');


// Credentials
const trainingKey = "14d248ab78ec4b82b0e08f17793e561a";
const predictionKey = "5761f6c8e081475b946c0ac3e13e3069";
const predictionResourceId = "/subscriptions/86640216-ba50-48ea-8c1c-bdc3d1d718ef/resourceGroups/GDC-RG-CognitiveServicesCustomVision/providers/Microsoft.CognitiveServices/accounts/GDCCustomVisionRes-Prediction";
const endPoint = "https://gdc-customvision-res.cognitiveservices.azure.com/"

const connectionStr = "DefaultEndpointsProtocol=https;AccountName=gdccustomvisionstorage;AccountKey=lm7hlNgCosAgULMah6c3nMEOhE4fr8t1WzSjhdJKNoBIYwbE5CzL7lL7/3AkCB88Jrq8++jpyAchzSZeib/nsA==;EndpointSuffix=core.windows.net"

// Authentication
const credentials = new msRest.ApiKeyCredentials({ inHeader: { "Training-key": trainingKey } });
const trainer = new TrainingApi.TrainingAPIClient(credentials, endPoint);
const predictor_credentials = new msRest.ApiKeyCredentials({ inHeader: { "Prediction-key": predictionKey } });
const predictor = new PredictionApi.PredictionAPIClient(predictor_credentials, endPoint);

// Utils
const setTimeoutPromise = util.promisify(setTimeout);
const longTOut = 10000;
const shortTOut = 200;

// Functions


async function createProject(trainer){
    
    const projectName = "ImageClassification-project-js"

    // Check if project already exists
    const projects = await trainer.getProjects()

    let project = null;

    for(const proj of projects){
        if(proj.name == projectName){
            console.log("Project already exists")
            project = await trainer.getProject(proj.id)
        }
    }

    // for(i=0; i<projects.length; i++){
    //     if(projects[i].name == projectName){
    //         console.log("Project already exists")
    //         project = await trainer.getProject(projects[i].id)
    //     }
    // }

    // If project doesn't exist create new one and upload images to Custom Vision
    let tags;
    if(project == null){
        const domains = await trainer.getDomains();
        const domain = domains.find(domain => domain.type === "Classification");
        project = await trainer.createProject(projectName, { "domainId": domain.id, "classificationType": "Multiclass" });
        
        // Create tags
        const crackedTag = await trainer.createTag(project.id, "cracked", {type: 'Regular'});
        const uncrackedTag = await trainer.createTag(project.id, "uncracked", {type:'Regular'});

        console.log("Project created")

        // Upload images to Custom Vision
        console.log("Adding images...");
        await uploadImages(trainer, project, crackedTag, uncrackedTag)
        tags = [crackedTag, uncrackedTag]
    }
    else{
        tags = await trainer.getTags(project.id);
    }
    
    return {project: project, tags: tags}

}

async function uploadImages(trainer, project, crackedTag, uncrackedTag){

    if(project != null){
        
        // Connect to blob storage
        const blobSClient = BlobServiceClient.fromConnectionString(connectionStr);
        const containerName = "train";
        const containerClient = blobSClient.getContainerClient(containerName);
    
        // List blobs' url
        let imageUrls = [];
        for await (const blob of containerClient.listBlobsFlat()) {
                imageUrls.push(containerClient.url + "/" + blob.name);
        }

        // Create list of images' url
        let imagesList = [];
        for(const url of imageUrls){
            if(!url.includes("bboxes.json")){
                if(url.includes("Cracked")){
                    const imageUrlEntry = {url: url, tagIds:[crackedTag.id]};
                    imagesList.push(imageUrlEntry);
                }
                else if(url.includes("Uncracked")){
                    const imageUrlEntry = {url: url, tagIds:[uncrackedTag.id]};
                    imagesList.push(imageUrlEntry);
                }
            }
        }
        
        // Upload images in batch
        const batchSize = 64;
        let i = 0;
        while(i < imagesList.length){
            const part = imagesList.slice(i, i + batchSize);
            const batch = {images: part};

            await trainer.createImagesFromUrls(project.id, batch);
            await setTimeoutPromise(shortTOut, null);
            
            i = i + batchSize;
        }
    }
}

async function trainModel(trainer, project){

    let trainingIteration = await trainer.trainProject(project.id);

    // Wait for training to complete
    console.log("Training started...");
    while (trainingIteration.status == "Training") {
        // console.log("Training status: " + trainingIteration.status);
        await setTimeoutPromise(longTOut, null);
        trainingIteration = await trainer.getIteration(project.id, trainingIteration.id);
    }
    console.log("Training status: " + trainingIteration.status);
    

    // Publish iteration
    const publishIterationName = "Iteration0";
    await trainer.publishIteration(project.id, trainingIteration.id, publishIterationName, predictionResourceId);
    console.log("Iteration published");

    return publishIterationName
}

async function testModel(predictor, project, tags, iterationName){

    // Parse tags 
    // for(tag of tags){
    //     if(tag.name == "cracked"){
    //         const crackedTag = tag;
    //    }
    //    else if(tag.name == "uncracked"){
    //         const uncrackedTag = tag;
    //    }
    // }

    // Create list of images' url
    const blobSClient = BlobServiceClient.fromConnectionString(connectionStr);
    const containerName = "test";
    const containerClient = blobSClient.getContainerClient(containerName);

    let imageUrls = [];
    for await (const blob of containerClient.listBlobsFlat()){
        imageUrls.push(containerClient.url + "/" + blob.name);
    }

    // Execute inference
    let resDict = {};
    for(const url of imageUrls){
        if(!url.includes("bboxes.json")){
            const result = await predictor.classifyImageUrl(project.id, iterationName, {url: url});

            await setTimeoutPromise(shortTOut, null);

            const splitUrl = url.split("/");
            const imageName = splitUrl[splitUrl.length-1];
            const imageClass = splitUrl[splitUrl.length-2].toLowerCase();

            resDict[imageName] = {
                "label": imageClass,
                "prediction": result.predictions[0].tagName,
                "confidence": result.predictions[0].probability * 100.0
            }
        }
    }
    
    return resDict;
}


function evalModel(resDict){

    let trueLabels = [];
    let predLabels = [];
    for(const [key, value] of Object.entries(resDict)){
        trueLabels.push(value["label"]);
        predLabels.push(value["prediction"]);
    }

    const confMatrix = ConfusionMatrix.fromLabels(trueLabels, predLabels);

    let metrics = {
        "accuracy": confMatrix.getAccuracy()
    }

    for(const label of confMatrix.getLabels()){
        metrics[label] = {
            "precision": confMatrix.getTruePositiveCount(label)/(confMatrix.getTruePositiveCount(label) + confMatrix.getFalsePositiveCount(label)),
            "recall": confMatrix.getTruePositiveCount(label)/(confMatrix.getTruePositiveCount(label) + confMatrix.getFalseNegativeCount(label))
        }
            
    }

    return metrics;
}


(async () => {

    // Create project
    console.log("Creating project...");
    let projectInfo = await createProject(trainer);
    
    // Train model
    console.log("Training model...");
    let iterationName = await trainModel(trainer, projectInfo.project);

    // Test model
    console.log("Testing model...");
    let results = await testModel(predictor, projectInfo.project, projectInfo.tags, iterationName);

    // Evaluate model
    console.log("Evaluating model...")
    let metrics = evalModel(results)
    console.log(metrics)

})()