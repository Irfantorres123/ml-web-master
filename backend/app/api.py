import glob
import os
import torch
import torch.nn.functional as F
import PIL.Image as Image
from model import Model
from generate_curve_spectrum_one_img import pipline
from torchvision import transforms
from cv2 import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import torchvision.transforms as tt
import numpy as np
import base64
from fastapi.staticfiles import StaticFiles
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
# origins = [
#     "http://localhost:3000",
#     "localhost:3000"
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

dir = './app'

#mnist get result
@app.get("/api/result")
async def result():
    model = Model(3,1)
    print(os.getcwd())
    model.load_state_dict(torch.load(os.path.join(dir,'./brain-mri-unet_cropped_30_epochs.pth')))
    filelist = glob.glob(os.path.join(dir,'./data/*'))
    trans = transforms.Compose([transforms.ToTensor()])
    ret = []
    i = 0
    if len(filelist) == 0:
        raise HTTPException(status_code=404, detail="input not found")
    for file in filelist:
        # preprocess and feed to model
        _, name = os.path.split(file)
        try:
            file = trans(Image.open(file))
            file = file.unsqueeze(0)  # type: ignore
            print(file.shape)
            outputs = model(file)
            outputs = F.softmax(outputs, dim=1)
            top_p, top_class = outputs.topk(5, dim=1)
            top_class = top_class[0]
            top_p = top_p[0]
        except:
            raise HTTPException(status_code=415, detail="unsupported input")
        print(top_class)
        print(top_p)
        dict = {}
        dict['key'] = str(i)
        dict['dir'] = 'api/getImage/'+name
        dict['name'] = name
        dict['pred1'] = str(top_class[0].item())
        dict['pred2'] = str(top_class[1].item())
        dict['pred3'] = str(top_class[2].item())
        dict['confi1'] = str(round(top_p[0].item(), 4))
        dict['confi2'] = str(round(top_p[1].item(), 4))
        dict['confi3'] = str(round(top_p[2].item(), 4))
        ret.append(dict)
        i += 1
    return ret
def get_encoded_image(image):
    image = cv2.imencode('.png', image)[1]
    image = base64.b64encode(image)
    return image

def cropped_image(image):
    image[np.nonzero(image<3)] = 0
    image[np.nonzero(image>=3)] = 1
    top=0
    bottom=image.shape[0]
    left=0
    right=image.shape[1]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j]==1:
                top=i
                break
        if top!=0:
            break
    for i in range(image.shape[0]-1,-1,-1):
        for j in range(image.shape[1]):
            if image[i][j]==1:
                bottom=i
                break
        if bottom!=image.shape[0]:
            break
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            if image[i][j]==1:
                left=j
                break
        if left!=0:
            break
    for j in range(image.shape[1]-1,-1,-1):
        for i in range(image.shape[0]):
            if image[i][j]==1:
                right=j
                break
        if right!=image.shape[1]:
            break
    top=max(0,top-10)
    bottom=min(bottom+10,image.shape[0])
    left=max(0,left-10)
    right=min(right+10,image.shape[1])    
    return image[top:bottom,left:right]*255

def concat(image,invert=False):
    final_image=np.array([])
    rows=[0]*2
    for i in range(2):
        groups=[0]*2
        for j in range(2):
            groups[j]=np.concatenate((image[j]*2,image[j]*2+1),axis=invert&1)
        row=np.concatenate((groups[0],groups[1]),axis=invert&1)
        rows[i]=row
    final_image=np.concatenate((rows[0],rows[1]),axis=0 if invert else 1)
    final_image.resize(256,256)
    return final_image
# scoliosis get result
@app.get("/api/result-prediction")
async def result_scoliosis():
    print("Reached")
    model=Model(3,1)
    model.load_state_dict(torch.load(os.path.join(dir,'./brain-mri-unet.pth'),map_location=torch.device('cpu')))
    filelist = glob.glob(os.path.join(dir,'data/*'))
    filelist = [i for i in filelist if '.tif' in i]
    print(filelist)
    ret = []
    i = 0
    if len(filelist) == 0:
        raise HTTPException(status_code=404, detail="input not found")
    for file in filelist:
        print(file)
        try:
            fileObj=cv2.imread(file)
            _, name = os.path.split(file)
            image = cv2.resize(fileObj, (128, 128))
            pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2)
            pred = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(pred)
            pred = model(pred.to(device))
            pred = pred.detach().cpu().numpy()[0,0,:,:]
            pred=pred*1000
            pred_t = np.copy(pred)
            print(pred_t)
            pred_t[np.nonzero(pred_t < 0.003)] = 0.0
            pred_t[np.nonzero(pred_t >= 0.003)] = 255.0
            pred_t = pred_t.astype("uint8") 

            pred_image=get_encoded_image(pred)
            pred_t_image=get_encoded_image(pred_t)
            original_image=get_encoded_image(fileObj)
            cropped=get_encoded_image(cropped_image(pred))
            probability=np.average(pred)/255*5*100
            probability=95 if probability>=95 else probability
            return {
                "original":original_image,
                "prediction":pred_image,
                "prediction_t":pred_t_image,
                "probability":probability,
                "cropped_image":cropped
            }
        except Exception as e:
            print(e)
            raise HTTPException(status_code=415, detail="unsupported input")
       
    return ret

# delete all uploaded files
@app.get("/api/clear")
async def clear():
    filelist = glob.glob(os.path.join(dir, "data/*"))
    secondfilelist=glob.glob(os.path.join(dir, "results/*"))
    for f in filelist:
        os.remove(f)
    for f in secondfilelist:
        os.remove(f)
    return 'success'

# return image path for display
@app.get('/api/getImage/{name}')
async def getImage(name: str):
    return FileResponse(path=os.path.join(dir, 'data/'+name))

# upload image
@app.post("/api/upload")
async def upload(file: UploadFile = File()):
    try:
        contents = await file.read()
        with open(os.path.join(dir, 'data/'+file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file(s)"}
    finally:
        await file.close()

    return {"message": f"Successfuly uploaded {file.filename}", "status":"success"}
            # x=32
            # y=32
            # cropped_images=[]
            # output=[]
            # output_t=[]
            # img=image.copy()
            # for i in range (4):
            #     t1=i*x
            #     for j in range(4):       
            #         t2=j*y             
            #         crop1 = img[t1:t1 + x, t2:t2 + y]
            #         image=cv2.resize(crop1,(128, 128),interpolation=cv2.INTER_LINEAR)
            #         # image = torch.unsqueeze(image, axis=0)
            #         # mask = torch.unsqueeze(mask, axis=0)
            #         cropped_images.append(image)
            #         image = np.expand_dims(image, axis=0)

            #         # pred = torch.tensor(image.astype(np.float32) / 255.)
            #         # print(pred.shape)
            #         pred = torch.tensor(image.astype(np.float32) / 255.).permute(0,3,1,2)
            #         # print(pred.shape)
            #         pred = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(pred)
            #         pred = model(pred.to(device))
            #         pred = pred.detach().cpu().numpy()[0,0,:,:]
            #         output.append(pred*100)
                    
            #         pred_t = np.copy(pred)
            #         pred_t[np.nonzero(pred_t < 0.3)] = 0.0
            #         pred_t[np.nonzero(pred_t >= 0.3)] = 255.
            #         pred_t = pred_t.astype("uint8")
            #         output_t.append(pred_t)
            # mn=np.mean(output_t)
            # final_image=get_encoded_image(concat(output))
            # final_image_t=get_encoded_image(concat(output_t))
            # original_image=get_encoded_image(img)
            # encoded_outputs=[]
            # for i in range(len(output_t)):
            #     encoded_outputs.append(get_encoded_image(output[i]))
            # print(mn/255*100," %")    
            # d={}
            # d['original']=original_image
            # d['predictions']=encoded_outputs
            # return d

app.mount('/public', StaticFiles(directory=os.path.join(dir,'../../client/build/')), name='public')

#serve react app at frontend/build
@app.get("/")
async def main():
    print("here")
    return FileResponse(path=os.path.join(dir, '../../client/build/index.html'))