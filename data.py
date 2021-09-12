def test3():
    patch_size = 25
    path = './dynamic_gesture_val.csv'
    df = pd.read_csv(path)
    Z_tr = []
    dfClass = df[df['class']=='left']

    Z_tr = np.expand_dims(np.array(dfClass[0:patch_size]), 0)

    for i in range(patch_size, dfClass.iloc[-1].name+1, patch_size):
        print([i, i+patch_size])
        Z_tr = np.append(Z_tr, np.expand_dims(np.array(dfClass[i:i+patch_size]), 0), axis=0)
    
    Z_tr = np.delete(Z_tr, [0, 1], 2).astype(np.float32)
    print(Z_tr)
    print(Z_tr.shape)

def test4():
    X_tr=[]          # variable to store entire dataset
    Y_tr=[]
  
    class_list = ['left', 'right']
    
    img_rows,img_cols=64,64

    from tqdm import tqdm
    csv_path = os.path.join("../dataset/dynamic/dynamic_gesture.csv")
    df = pd.read_csv(csv_path)
    ls_path = os.path.join("../dataset/dynamic/rgb/left")

    listing = os.listdir(ls_path)
    #csv_listing = os.listdir(csv_path)

    frames = []
    img_label = []
    img_depth=1
    for imgs in listing:
        n = imgs[:imgs.find('_')]
        img = os.path.join(ls_path, imgs)
        frame = cv2.imread(img)
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_label.append(class_list[0])
        if int(n) != img_depth:
            #print('change')
            img_depth=img_depth+1
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)
            frames = []
            Y_tr.append(img_label)
            img_label=[]
      
        frames.append(gray)

    input_img = np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    X_tr.append(ipt)
    print(ipt.shape)
    print(len(X_tr))