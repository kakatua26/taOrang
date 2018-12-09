def mainDWT(nilaiK):
    import pywt
    import cv2
    import numpy as np
    import math
    import cmath
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.cross_validation import train_test_split #memisahkan data latih & data uji
    from sklearn.metrics import accuracy_score #menghitung akurasi

    #training data

    standar_dLL = []
    standar_dLH = []
    standar_dHL = []
    standar_dHH = []
    mean_LL_list = []
    mean_LH_list = []
    mean_HL_list = []
    mean_HH_list = []

    standar_dLL1 = []
    standar_dLH1 = []
    standar_dHL1 = []
    standar_dHH1 = []
    mean_LL1_list = []
    mean_LH1_list = []
    mean_HL1_list = []
    mean_HH1_list = []
    group = []
    for i in range(1,73):
        img = cv2.imread('DATA/tangan%d.jpg' %(i)) #load gambar ke-i
        resized_image = cv2.resize(img, (500,256)) #rezise image
        img_n = (cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)) #ubah gambar ke grayscale

        coeffs2 = pywt.dwt2(img_n, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        LL1, (LH1, HL1, HH1) = pywt.dwt2(LL, 'bior1.3')
        LLr = LL.ravel()
        LHr = LH.ravel()
        HLr = HL.ravel()
        HHr = HH.ravel()
        LL1r = LL1.ravel()
        LH1r = LH1.ravel()
        HL1r = HL1.ravel()
        HH1r = HH1.ravel()

        N = len(LLr)
        N1 = len(LL1r)
        
        #mean LL
        mean_LL = sum(LLr)/N
        #print("mean LL %d: " %(i), mean_LL)
        mean_LL_list.append(mean_LL)
        
        #mean LH
        mean_LH = sum(LHr)/N
        #print("mean LH %d: " %(i), mean_LH)
        mean_LH_list.append(mean_LH)
        
        #mean HL
        mean_HL = sum(HLr)/N
        #print("mean HL %d: " %(i), mean_HL)
        mean_HL_list.append(mean_HL)
        
        #mean HH
        mean_HH = sum(HHr)/N
        #print("mean HH %d: " %(i), mean_HH)
        mean_HH_list.append(mean_HH)
        
        
        #standar_d LL
        LL_list = []
        for j in LLr:
            LLmean = (j - mean_LL)*(j - mean_LL)
            LL_list.append(LLmean)
        xi2LL = sum(LL_list)
        varianLL = xi2LL/N
        standar_deviasiLL = cmath.sqrt(varianLL)
        #print("standar deviasi LL %d: " %(i), standar_deviasiLL)
        standar_dLL.append(standar_deviasiLL)
        
        #standar_d LH
        LH_list = []
        for k in LHr:
            LHmean = (k - mean_LH)*(k - mean_LH)
            LH_list.append(LHmean)
        xi2LH = sum(LH_list)
        varianLH = xi2LH/N
        standar_deviasiLH = cmath.sqrt(varianLH)
        #print("standar deviasi LH %d: " %(i), standar_deviasiLH)
        standar_dLH.append(standar_deviasiLH)
        
        #standar_d HL
        HL_list = []
        for l in HLr:
            HLmean = (l - mean_HL)*(l - mean_HL)
            HL_list.append(HLmean)
        xi2HL = sum(HL_list)
        varianHL = xi2HL/N
        standar_deviasiHL = cmath.sqrt(varianHL)
        #print("standar deviasi HL %d: " %(i), standar_deviasiHL)
        standar_dHL.append(standar_deviasiHL)
        
        #standar_d HH
        HH_list = []
        for m in HHr:
            HHmean = (m - mean_HH)*(m - mean_HH)
            HH_list.append(HHmean)
        xi2HH = sum(HH_list)
        varianHH = xi2HH/N
        standar_deviasiHH = cmath.sqrt(varianHH)
        #print("standar deviasi HH %d: " %(i), standar_deviasiHH)
        standar_dHH.append(standar_deviasiHH)
        
        
        #mean LL1
        mean_LL1 = sum(LL1r)/N1
        #print("mean LL1 %d: " %(i), mean_LL1)
        mean_LL1_list.append(mean_LL1)
        
        #mean LH1
        mean_LH1 = sum(LH1r)/N1
        #print("mean LH1 %d: " %(i), mean_LH1)
        mean_LH1_list.append(mean_LH1)
        
        #mean HL1
        mean_HL1 = sum(HL1r)/N1
        #print("mean HL1 %d: " %(i), mean_HL1)
        mean_HL1_list.append(mean_HL1)
        
        #mean HH1
        mean_HH1 = sum(HH1r)/N1
        #print("mean HH1 %d: " %(i), mean_HH1)
        mean_HH1_list.append(mean_HH1)
        
        
        #standar_d LL1
        LL1_list = []
        for n in LL1r:
            LL1mean = (n - mean_LL1)*(n - mean_LL1)
            LL1_list.append(LL1mean)
        xi2LL1 = sum(LL1_list)
        varianLL1 = xi2LL1/N1
        standar_deviasiLL1 = cmath.sqrt(varianLL1)
        #print("standar deviasi LL1 %d: " %(i), standar_deviasiLL1)
        standar_dLL1.append(standar_deviasiLL1)
        
        #standar_d LH1
        LH1_list = []
        for o in LH1r:
            LH1mean = (o - mean_LH1)*(o - mean_LH1)
            LH1_list.append(LH1mean)
        xi2LH1 = sum(LH1_list)
        varianLH1 = xi2LH1/N1
        standar_deviasiLH1 = cmath.sqrt(varianLH1)
        #print("standar deviasi LH1 %d: " %(i), standar_deviasiLH1)
        standar_dLH1.append(standar_deviasiLH1)
        
        #standar_d HL1
        HL1_list = []
        for p in HL1r:
            HL1mean = (p - mean_HL1)*(p - mean_HL1)
            HL1_list.append(HL1mean)
        xi2HL1 = sum(HL1_list)
        varianHL1 = xi2HL1/N1
        standar_deviasiHL1 = cmath.sqrt(varianHL1)
        #print("standar deviasi HL1 %d: " %(i), standar_deviasiHL1)
        standar_dHL1.append(standar_deviasiHL1)
        
        #standar_d HH1
        HH1_list = []
        for q in HH1r:
            HH1mean = (q - mean_HH1)*(q - mean_HH1)
            HH1_list.append(HH1mean)
        xi2HH1 = sum(HH1_list)
        varianHH1 = xi2HH1/N1
        standar_deviasiHH1 = cmath.sqrt(varianHH1)
        #print("standar deviasi HH1 %d: " %(i), standar_deviasiHH1)
        standar_dHH1.append(standar_deviasiHH1)
        
        
        
        if (i>=1 and i<=34):
            group.append("Terdeteksi")
        else:
            group.append("Tidak Terdeteksi")


    ########################################################################

    #testing data

    standar_dLL_test = []
    standar_dLH_test = []
    standar_dHL_test = []
    standar_dHH_test = []
    mean_LL_list_test = []
    mean_LH_list_test = []
    mean_HL_list_test = []
    mean_HH_list_test = []

    standar_dLL1_test = []
    standar_dLH1_test = []
    standar_dHL1_test = []
    standar_dHH1_test = []
    mean_LL1_list_test = []
    mean_LH1_list_test = []
    mean_HL1_list_test = []
    mean_HH1_list_test = []
    group_test = []
    for i in range(1,9):
        img = cv2.imread('DATA/tangantest%d.jpg' %(i)) #load gambar ke-i
        resized_image = cv2.resize(img, (500,256)) #rezise image
        img_n = (cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)) #ubah gambar ke grayscale

        coeffs2 = pywt.dwt2(img_n, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        LL1, (LH1, HL1, HH1) = pywt.dwt2(LL, 'bior1.3')
        LLr = LL.ravel()
        LHr = LH.ravel()
        HLr = HL.ravel()
        HHr = HH.ravel()
        LL1r = LL1.ravel()
        LH1r = LH1.ravel()
        HL1r = HL1.ravel()
        HH1r = HH1.ravel()


        N = len(LLr)
        N1 = len(LL1r)
        
        #mean LL
        mean_LL = sum(LLr)/N
        #print("mean LL %d: " %(i), mean_LL)
        mean_LL_list_test.append(mean_LL)
        
        #mean LH
        mean_LH = sum(LHr)/N
        #print("mean LH %d: " %(i), mean_LH)
        mean_LH_list_test.append(mean_LH)
        
        #mean HL
        mean_HL = sum(HLr)/N
        #print("mean HL %d: " %(i), mean_HL)
        mean_HL_list_test.append(mean_HL)
        
        #mean HH
        mean_HH = sum(HHr)/N
        #print("mean HH %d: " %(i), mean_HH)
        mean_HH_list_test.append(mean_HH)
        
        #standar_d LL
        LL_list = []
        for j in LLr:
            LLmean = (j - mean_LL)*(j - mean_LL)
            LL_list.append(LLmean)
        xi2LL = sum(LL_list)
        varianLL = xi2LL/N
        standar_deviasiLL = cmath.sqrt(varianLL)
        #print("standar deviasi LL %d: " %(i), standar_deviasiLL)
        standar_dLL_test.append(standar_deviasiLL)
        
        #standar_d LH
        LH_list = []
        for k in LHr:
            LHmean = (k - mean_LH)*(k - mean_LH)
            LH_list.append(LHmean)
        xi2LH = sum(LH_list)
        varianLH = xi2LH/N
        standar_deviasiLH = cmath.sqrt(varianLH)
        #print("standar deviasi LH %d: " %(i), standar_deviasiLH)
        standar_dLH_test.append(standar_deviasiLH)
        
        #standar_d HL
        HL_list = []
        for l in HLr:
            HLmean = (l - mean_HL)*(l - mean_HL)
            HL_list.append(HLmean)
        xi2HL = sum(HL_list)
        varianHL = xi2HL/N
        standar_deviasiHL = cmath.sqrt(varianHL)
        #print("standar deviasi HL %d: " %(i), standar_deviasiHL)
        standar_dHL_test.append(standar_deviasiHL)
        
        #standar_d HH
        HH_list = []
        for m in HHr:
            HHmean = (m - mean_HH)*(m - mean_HH)
            HH_list.append(HHmean)
        xi2HH = sum(HH_list)
        varianHH = xi2HH/N
        standar_deviasiHH = cmath.sqrt(varianHH)
        #print("standar deviasi HH %d: " %(i), standar_deviasiHH)
        standar_dHH_test.append(standar_deviasiHH)
        
        
        
        #mean LL1
        mean_LL1 = sum(LL1r)/N1
        #print("mean LL %d: " %(i), mean_LL)
        mean_LL1_list_test.append(mean_LL1)
        
        #mean LH1
        mean_LH1 = sum(LH1r)/N1
        #print("mean LH %d: " %(i), mean_LH)
        mean_LH1_list_test.append(mean_LH1)
        
        #mean HL1
        mean_HL1 = sum(HL1r)/N1
        #print("mean HL %d: " %(i), mean_HL)
        mean_HL1_list_test.append(mean_HL1)
        
        #mean HH1
        mean_HH1 = sum(HH1r)/N1
        #print("mean HH %d: " %(i), mean_HH)
        mean_HH1_list_test.append(mean_HH1)
        
        #standar_d LL1
        LL1_list = []
        for n in LL1r:
            LL1mean = (n - mean_LL1)*(n - mean_LL1)
            LL1_list.append(LL1mean)
        xi2LL1 = sum(LL1_list)
        varianLL1 = xi2LL1/N1
        standar_deviasiLL1 = cmath.sqrt(varianLL1)
        #print("standar deviasi LL %d: " %(i), standar_deviasiLL)
        standar_dLL1_test.append(standar_deviasiLL1)
        
        #standar_d LH1
        LH1_list = []
        for o in LH1r:
            LH1mean = (o - mean_LH1)*(o - mean_LH1)
            LH1_list.append(LH1mean)
        xi2LH1 = sum(LH1_list)
        varianLH1 = xi2LH1/N1
        standar_deviasiLH1 = cmath.sqrt(varianLH1)
        #print("standar deviasi LH %d: " %(i), standar_deviasiLH)
        standar_dLH1_test.append(standar_deviasiLH1)
        
        #standar_d HL1
        HL1_list = []
        for p in HL1r:
            HL1mean = (p - mean_HL1)*(p - mean_HL1)
            HL1_list.append(HL1mean)
        xi2HL1 = sum(HL1_list)
        varianHL1 = xi2HL1/N1
        standar_deviasiHL1 = cmath.sqrt(varianHL1)
        #print("standar deviasi HL %d: " %(i), standar_deviasiHL)
        standar_dHL1_test.append(standar_deviasiHL1)
        
        #standar_d HH1
        HH1_list = []
        for q in HH1r:
            HH1mean = (q - mean_HH1)*(q - mean_HH1)
            HH1_list.append(HH1mean)
        xi2HH1 = sum(HH1_list)
        varianHH1 = xi2HH1/N1
        standar_deviasiHH1 = cmath.sqrt(varianHH1)
        #print("standar deviasi HH %d: " %(i), standar_deviasiHH)
        standar_dHH1_test.append(standar_deviasiHH1)
        
        if (i>=1 and i<=4):
            group_test.append("Terdeteksi")
        else:
            group_test.append("Tidak Terdeteksi")
            


    ##########################################################


    #data frame fitur

    dLL = {'standar deviasi LL': standar_dLL,
           'mean LL': mean_LL_list}
    dLH = {'standar deviasi LH': standar_dLH,
           'mean LH': mean_LH_list}
    dHL = {'standar deviasi HL': standar_dHL,
           'mean HL': mean_HL_list}
    dHH = {'standar deviasi HH': standar_dHH,
           'mean HH': mean_HH_list}
    dLL1 = {'standar deviasi LL1': standar_dLL1,
           'mean LL1': mean_LL1_list}
    dLH1 = {'standar deviasi LH1': standar_dLH1,
           'mean LH1': mean_LH1_list}
    dHL1 = {'standar deviasi HL1': standar_dHL1,
           'mean HL1': mean_HL1_list}
    dHH1 = {'standar deviasi HH1': standar_dHH1,
           'mean HH1': mean_HH1_list}

    dLL_test = {'standar deviasi LL _test': standar_dLL_test,
           'mean LL _test': mean_LL_list_test}
    dLH_test = {'standar deviasi LH _test': standar_dLH_test,
           'mean LH _test': mean_LH_list_test}
    dHL_test = {'standar deviasi HL _test': standar_dHL_test,
           'mean HL': mean_HL_list_test}
    dHH_test = {'standar deviasi HH _test': standar_dHH_test,
           'mean HH _test': mean_HH_list_test}
    dLL1_test = {'standar deviasi LL1 _test': standar_dLL1_test,
           'mean LL1 _test': mean_LL1_list_test}
    dLH1_test = {'standar deviasi LH1 _test': standar_dLH1_test,
           'mean LH1 _test': mean_LH1_list_test}
    dHL1_test = {'standar deviasi HL1 _test': standar_dHL1_test,
           'mean HL1': mean_HL1_list_test}
    dHH1_test = {'standar deviasi HH1 _test': standar_dHH1_test,
           'mean HH1 _test': mean_HH1_list_test}


    dall = {'standar deviasi LL': standar_dLL,
           'mean LL': mean_LL_list,
            'standar deviasi LH': standar_dLH,
           'mean LH': mean_LH_list,
            'standar deviasi HL': standar_dHL,
           'mean HL': mean_HL_list,
            'standar deviasi HH': standar_dHH,
           'mean HH': mean_HH_list,
            'standar deviasi LL1': standar_dLL1,
           'mean LL1': mean_LL1_list,
            'standar deviasi LH1': standar_dLH1,
           'mean LH1': mean_LH1_list,
            'standar deviasi HL1': standar_dHL1,
           'mean HL1': mean_HL1_list,
            'standar deviasi HH1': standar_dHH1,
           'mean HH1': mean_HH1_list}

    dall_test = {'standar deviasi LL _test': standar_dLL_test,
                 'mean LL _test': mean_LL_list_test,
                 'standar deviasi LH _test': standar_dLH_test,
                 'mean LH _test': mean_LH_list_test,
                 'standar deviasi HL _test': standar_dHL_test,
                 'mean HL': mean_HL_list_test,
                 'standar deviasi HH _test': standar_dHH_test,
                 'mean HH _test': mean_HH_list_test,
                 'standar deviasi LL1 _test': standar_dLL1_test,
                 'mean LL1 _test': mean_LL1_list_test,
                 'standar deviasi LH1 _test': standar_dLH1_test,
                 'mean LH1 _test': mean_LH1_list_test,
                 'standar deviasi HL1 _test': standar_dHL1_test,
                 'mean HL1': mean_HL1_list_test,
                 'standar deviasi HH1 _test': standar_dHH1_test,
                 'mean HH1 _test': mean_HH1_list_test}

    dfLL = pd.DataFrame(dLL)
    dfLH = pd.DataFrame(dLH)
    dfHL = pd.DataFrame(dHL)
    dfHH = pd.DataFrame(dHH)
    dfLL1 = pd.DataFrame(dLL1)
    dfLH1 = pd.DataFrame(dLH1)
    dfHL1 = pd.DataFrame(dHL1)
    dfHH1 = pd.DataFrame(dHH1)

    dfLL_test = pd.DataFrame(dLL_test)
    dfLH_test = pd.DataFrame(dLH_test)
    dfHL_test = pd.DataFrame(dHL_test)
    dfHH_test = pd.DataFrame(dHH_test)
    dfLL1_test = pd.DataFrame(dLL1_test)
    dfLH1_test = pd.DataFrame(dLH1_test)
    dfHL1_test = pd.DataFrame(dHL1_test)
    dfHH1_test = pd.DataFrame(dHH1_test)

    dfall = pd.DataFrame(dall)
    dfall_test = pd.DataFrame(dall_test)


    dfLLarray = dfLL.values
    dfLHarray = dfLH.values
    dfHLarray = dfHL.values
    dfHHarray = dfHH.values
    dfLL1array = dfLL1.values
    dfLH1array = dfLH1.values
    dfHL1array = dfHL1.values
    dfHH1array = dfHH1.values

    dfLLarray_test = dfLL_test.values
    dfLHarray_test = dfLH_test.values
    dfHLarray_test = dfHL_test.values
    dfHHarray_test = dfHH_test.values
    dfLL1array_test = dfLL1_test.values
    dfLH1array_test = dfLH1_test.values
    dfHL1array_test = dfHL1_test.values
    dfHH1array_test = dfHH1_test.values

    dfallarray_test = dfall_test.values
    dfallarray = dfall.values



    groups = np.asarray(group)
    groups_test = np.asarray(group_test)

    ##########################################################

    #proses klasifikasi
    k = nilaiK
    X_train = dfallarray
    y_train = groups
    X_test = dfallarray_test
    y_test = groups_test

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    prediksi = neigh.predict(X_test)
    akurasi = accuracy_score(prediksi, y_test)
    #print(prediksi)
    #print(round(akurasi*100,2))

    return prediksi

if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
