import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import decomposition
from scipy import stats
from sklearn import cluster


ROOT_DIR =  os.path.abspath('./')
sys.path.append(ROOT_DIR)

matplotlib.style.use('fivethirtyeight')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10,10)
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams["font.family"] = 'NanumGothic'

count = 1
output = {}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def countFile(key):
    if count < 10:
        key_num = "0{}".format(count)
    else:
        key_num = "{}".format(count)
    value = "{}.png".format(key)
    output.update({str(key_num):value})
    count = count+1
    

def ConvertToImageCoords(latCoord, longCoord, latRange, longRange, imageSize):
    latInds  = imageSize[0] - (imageSize[0] * (latCoord  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
    longInds = (imageSize[1] * (longCoord - longRange[0]) / (longRange[1] - longRange[0])).astype(int)
    return latInds, longInds


def run(file,path):
    # dataDir = '../input/'
    taxiDB = pd.read_csv(os.path.join(ROOT_DIR,'all_data.csv'))

    # remove obvious outliers
    allLat  = np.array(list(taxiDB['pickup_latitude'])  + list(taxiDB['dropoff_latitude']))
    allLong = np.array(list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude']))

    longLimits = [np.percentile(allLong, 0.3), np.percentile(allLong, 99.7)]
    latLimits  = [np.percentile(allLat , 0.3), np.percentile(allLat , 99.7)]
    durLimits  = [np.percentile(taxiDB['trip_duration'], 0.4), np.percentile(taxiDB['trip_duration'], 99.7)]

    taxiDB = taxiDB[(taxiDB['pickup_latitude']   >= latLimits[0] ) & (taxiDB['pickup_latitude']   <= latLimits[1]) ]
    taxiDB = taxiDB[(taxiDB['dropoff_latitude']  >= latLimits[0] ) & (taxiDB['dropoff_latitude']  <= latLimits[1]) ]
    taxiDB = taxiDB[(taxiDB['pickup_longitude']  >= longLimits[0]) & (taxiDB['pickup_longitude']  <= longLimits[1])]
    taxiDB = taxiDB[(taxiDB['dropoff_longitude'] >= longLimits[0]) & (taxiDB['dropoff_longitude'] <= longLimits[1])]
    taxiDB = taxiDB[(taxiDB['trip_duration']     >= durLimits[0] ) & (taxiDB['trip_duration']     <= durLimits[1]) ]
    taxiDB = taxiDB.reset_index(drop=True)

    allLat  = np.array(list(taxiDB['pickup_latitude'])  + list(taxiDB['dropoff_latitude']))
    allLong = np.array(list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude']))

    # convert fields to sensible units
    medianLat  = np.percentile(allLat,50)
    medianLong = np.percentile(allLong,50)

    latMultiplier  = 111.32
    longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32

    taxiDB['duration [min]'] = taxiDB['trip_duration'] /60.0
    taxiDB['src lat [km]']   = latMultiplier  * (taxiDB['pickup_latitude']   - medianLat)
    taxiDB['src long [km]']  = longMultiplier * (taxiDB['pickup_longitude']  - medianLong)
    taxiDB['dst lat [km]']   = latMultiplier  * (taxiDB['dropoff_latitude']  - medianLat)
    taxiDB['dst long [km]']  = longMultiplier * (taxiDB['dropoff_longitude'] - medianLong)

    allLat  = np.array(list(taxiDB['src lat [km]'])  + list(taxiDB['dst lat [km]']))
    allLong = np.array(list(taxiDB['src long [km]']) + list(taxiDB['dst long [km]']))

        # make sure the ranges we chose are sensible
    fig, axArray = plt.subplots(nrows=1,ncols=3,figsize=(13,4))
    axArray[0].hist(taxiDB['duration [min]'],80); 
    axArray[0].set_xlabel('활동 기간 [분]'); axArray[0].set_ylabel('개수')
    axArray[1].hist(allLat ,80); axArray[1].set_xlabel('위도 [km]')
    axArray[2].hist(allLong,80); axArray[2].set_xlabel('경도 [km]')

    plt.savefig('{}/{}'.format(path,'histogram'))
    plt.clf()
    countFile('histogram')

        #%% plot scatter of trip duration vs. aerial distance between pickup and dropoff
    taxiDB['log duration']       = np.log1p(taxiDB['duration [min]'])
    taxiDB['euclidian distance'] = np.sqrt((taxiDB['src lat [km]']  - taxiDB['dst lat [km]'] )**2 + 
                                        (taxiDB['src long [km]'] - taxiDB['dst long [km]'])**2)

    fig, axArray = plt.subplots(nrows=1,ncols=2,figsize=(13,6))
    axArray[0].scatter(taxiDB['euclidian distance'], taxiDB['duration [min]'],c='r',s=5,alpha=0.01); 
    axArray[0].set_xlabel('두 지점간의 거리 [km]'); axArray[0].set_ylabel('기간 [분]')
    axArray[0].set_xlim(taxiDB['euclidian distance'].min(),taxiDB['euclidian distance'].max())
    axArray[0].set_ylim(taxiDB['duration [min]'].min(),taxiDB['duration [min]'].max())
    axArray[0].set_title('운행 기간 및 두 지점 간의 거리 상관관계')

    axArray[1].scatter(taxiDB['euclidian distance'], taxiDB['log duration'],c='r',s=5,alpha=0.01); 
    axArray[1].set_xlabel('두 지점간의 거리 [km]'); axArray[1].set_ylabel('log(1+거리) [log(분)]')
    axArray[1].set_xlim(taxiDB['euclidian distance'].min(),taxiDB['euclidian distance'].max())
    axArray[1].set_ylim(taxiDB['log duration'].min(),taxiDB['log duration'].max())
    axArray[1].set_title('운행 기간의 두 지점간의 역관계')

    plt.savefig('{}/{}'.format(path,'distance'))
    plt.clf()
    countFile('distance')
    
    # show the log density of pickup and dropoff locations
    imageSize = (700,700)
    longRange = [-400,400]
    latRange = [-400,400]
    # 700 - 700 * allLat - latRange[0]
    allLatInds  = imageSize[0] - (imageSize[0] * (allLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
    allLongInds = (imageSize[1] * (allLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)

    locationDensityImage = np.zeros(imageSize)
    for latInd, longInd in zip(allLatInds,allLongInds):
        locationDensityImage[latInd,longInd] += 1

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
    ax.imshow(np.log(locationDensityImage+1),cmap='hot')
    ax.set_axis_off()

    plt.savefig('{}/{}'.format(path,'visualize_map'))
    plt.clf()
    countFile('visualize_map')
    
    pickupTime = pd.to_datetime(taxiDB['pickup_datetime'])

    taxiDB['src hourOfDay'] = (pickupTime.dt.hour*60.0 + pickupTime.dt.minute)   / 60.0
    taxiDB['dst hourOfDay'] = taxiDB['src hourOfDay'] + taxiDB['duration [min]'] / 60.0

    taxiDB['dayOfWeek']     = pickupTime.dt.weekday
    taxiDB['hourOfWeek']    = taxiDB['dayOfWeek']*24.0 + taxiDB['src hourOfDay']

    taxiDB['monthOfYear']   = pickupTime.dt.month
    taxiDB['dayOfYear']     = pickupTime.dt.dayofyear
    taxiDB['weekOfYear']    = pickupTime.dt.weekofyear
    taxiDB['hourOfYear']    = taxiDB['dayOfYear']*24.0 + taxiDB['src hourOfDay']

    tripAttributes = np.array(taxiDB.loc[:,['src lat [km]','src long [km]','dst lat [km]','dst long [km]','duration [min]']])
    meanTripAttr = tripAttributes.mean(axis=0)
    stdTripAttr  = tripAttributes.std(axis=0)
    tripAttributes = stats.zscore(tripAttributes, axis=0)

    numClusters = 80
    TripKmeansModel = cluster.MiniBatchKMeans(n_clusters=numClusters, batch_size=120000, n_init=100, random_state=1)
    clusterInds = TripKmeansModel.fit_predict(tripAttributes)

    clusterTotalCounts, _ = np.histogram(clusterInds, bins=numClusters)
    sortedClusterInds = np.flipud(np.argsort(clusterTotalCounts))

    plt.figure(figsize=(12,4)); plt.title('모든 운행 기록 클러스터 히스토그램')
    plt.bar(range(1,numClusters+1),clusterTotalCounts[sortedClusterInds])
    plt.ylabel('빈도수 [개수]'); plt.xlabel('Cluster index (sorted by 클러스터 빈도수에 따라 정렬됨)')
    plt.xlim(0,numClusters+1)

    plt.savefig('{}/{}'.format(path,'histogram_cluster'))
    plt.clf()
    countFile('histogram_cluster')


    templateTrips = TripKmeansModel.cluster_centers_ * np.tile(stdTripAttr,(numClusters,1)) + np.tile(meanTripAttr,(numClusters,1))

    srcCoords = templateTrips[:,:2]
    dstCoords = templateTrips[:,2:4]

    srcImCoords = ConvertToImageCoords(srcCoords[:,0],srcCoords[:,1], latRange, longRange, imageSize)
    dstImCoords = ConvertToImageCoords(dstCoords[:,0],dstCoords[:,1], latRange, longRange, imageSize)

    plt.figure(figsize=(12,12))
    plt.imshow(np.log(locationDensityImage+1),cmap='hot')

    plt.grid(b=None)
    plt.scatter(srcImCoords[1],srcImCoords[0],c='m',s=200,alpha=0.8)
    plt.scatter(dstImCoords[1],dstImCoords[0],c='g',s=200,alpha=0.8)

    plt.savefig('{}/{}'.format(path,'show_map'))
    plt.clf()
    countFile('show_map')


    for i in range(len(srcImCoords[0])):
        plt.arrow(srcImCoords[1][i],srcImCoords[0][i], dstImCoords[1][i]-srcImCoords[1][i], dstImCoords[0][i]-srcImCoords[0][i], 
                edgecolor='c', facecolor='c', width=0.8,alpha=0.4,head_width=10.0,head_length=10.0,length_includes_head=True)

        # calculate the trip distribution for different hours of the weekday
    hoursOfDay = np.sort(taxiDB['src hourOfDay'].astype(int).unique())
    clusterDistributionHourOfDay_weekday = np.zeros((len(hoursOfDay),numClusters))
    for k, hour in enumerate(hoursOfDay):
        slectedInds = (taxiDB['src hourOfDay'].astype(int) == hour) & (taxiDB['dayOfWeek'] <= 4)
        currDistribution, _ = np.histogram(clusterInds[slectedInds], bins=numClusters)
        clusterDistributionHourOfDay_weekday[k,:] = currDistribution[sortedClusterInds]

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
    ax.set_title('평일 운행기록  ', fontsize=12)
    ax.imshow(clusterDistributionHourOfDay_weekday); ax.grid('off')
    ax.set_xlabel('운행기록 클러스터'); ax.set_ylabel('하루 시간')
    ax.annotate('심야 시간대', color='r', fontsize=15, xy=(52, 2), xytext=(58, 1.75),
                arrowprops=dict(facecolor='red', shrink=0.03))
   
    plt.savefig('{}/{}'.format(path,'trip_duration_weekday'))
    plt.clf()
    countFile('trip_duration_weekday')
        # calculate the trip distribution for different hours of the weekend
    hoursOfDay = np.sort(taxiDB['src hourOfDay'].astype(int).unique())
    clusterDistributionHourOfDay_weekend = np.zeros((len(hoursOfDay),numClusters))
    for k, hour in enumerate(hoursOfDay):
        slectedInds = (taxiDB['src hourOfDay'].astype(int) == hour) & (taxiDB['dayOfWeek'] >= 5)
        currDistribution, _ = np.histogram(clusterInds[slectedInds], bins=numClusters)
        clusterDistributionHourOfDay_weekend[k,:] = currDistribution[sortedClusterInds]

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
    ax.set_title('주말 운행 기록', fontsize=12)
    ax.imshow(clusterDistributionHourOfDay_weekend); ax.grid('off')
    ax.set_xlabel('운행 클러스터'); ax.set_ylabel('하루 시간')
    ax.annotate('심야', color='r', fontsize=15, xy=(52, 2), xytext=(58, 1.75),
                arrowprops=dict(facecolor='red', shrink=0.03))
    ax.annotate('늦은 아침 (주말)', color='r', fontsize=15, xy=(45, 10), xytext=(58, 9.75),
                arrowprops=dict(facecolor='red', shrink=0.03))
    
    plt.savefig('{}/{}'.format(path,'trip_duration_weekend'))
    plt.clf()
    countFile('trip_duration_weekend')
        # calculate the trip distribution for day of week
    daysOfWeek = np.sort(taxiDB['dayOfWeek'].unique())
    clusterDistributionDayOfWeek = np.zeros((len(daysOfWeek),numClusters))
    for k, day in enumerate(daysOfWeek):
        slectedInds = taxiDB['dayOfWeek'] == day
        currDistribution, _ = np.histogram(clusterInds[slectedInds], bins=numClusters)
        clusterDistributionDayOfWeek[k,:] = currDistribution[sortedClusterInds]

    plt.figure(figsize=(12,5)); plt.title('주중간의 운행 기록')
    plt.imshow(clusterDistributionDayOfWeek); plt.grid('off')
    plt.xlabel('운행 클러스터'); plt.ylabel('주중')
    
    plt.savefig('{}/{}'.format(path,'day_of_week'))
    plt.clf()
    countFile('day_of_week')


    # calculate the trip distribution for day of year
    daysOfYear = taxiDB['dayOfYear'].unique()
    daysOfYear = np.sort(daysOfYear)
    clusterDistributionDayOfYear = np.zeros((len(daysOfYear),numClusters))
    for k, day in enumerate(daysOfYear):
        slectedInds = taxiDB['dayOfYear'] == day
        currDistribution, _ = np.histogram(clusterInds[slectedInds], bins=numClusters)
        clusterDistributionDayOfYear[k,:] = currDistribution[sortedClusterInds]

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,16))
    ax.set_title('월간 운행 기록 분포', fontsize=12)
    ax.imshow(clusterDistributionDayOfYear); ax.grid('off')
    ax.set_xlabel('운행 클러스터'); ax.set_ylabel('한달 분포')

    plt.savefig('{}/{}'.format(path,'month_distribution'))
    plt.clf()
    countFile('month_distribution')
    hoursOfYear = np.sort(taxiDB['hourOfYear'].astype(int).unique())

    clusterDistributionHourOfYear = np.zeros((len(range(hoursOfYear[0],hoursOfYear[-1]+1)),numClusters))
    dayOfYearVec  = np.zeros(clusterDistributionHourOfYear.shape[0])
    weekdayVec    = np.zeros(clusterDistributionHourOfYear.shape[0])
    weekOfYearVec = np.zeros(clusterDistributionHourOfYear.shape[0])



    for k, hour in enumerate(hoursOfYear):
        slectedInds = taxiDB['hourOfYear'].astype(int) == hour
        currDistribution, _ = np.histogram(clusterInds[slectedInds], bins=numClusters)
        if k == len(clusterDistributionDayOfYear):
            k = k-1
        clusterDistributionHourOfYear[k,:] = currDistribution[sortedClusterInds]
        dayOfYearVec[k]  = taxiDB[slectedInds]['dayOfYear'].mean()
        weekdayVec[k]    = taxiDB[slectedInds]['dayOfWeek'].mean()
        weekOfYearVec[k] = taxiDB[slectedInds]['weekOfYear'].mean()

    numComponents = 3
    TripDistributionPCAModel = decomposition.PCA(n_components=numComponents,whiten=True, random_state=1)
    compactClusterDistributionHourOfYear = TripDistributionPCAModel.fit_transform(clusterDistributionHourOfYear)

    listOfFullWeeks = []
    for uniqueVal in np.unique(weekOfYearVec):
        if (weekOfYearVec == uniqueVal).sum() == 24*7:
            listOfFullWeeks.append(uniqueVal)

    weeklyTraces = np.zeros((24*7,numComponents,len(listOfFullWeeks)))
    for k, weekInd in enumerate(listOfFullWeeks):
        weeklyTraces[:,:,k] = compactClusterDistributionHourOfYear[weekOfYearVec == weekInd,:]

    fig, axArray = plt.subplots(nrows=numComponents,ncols=1,sharex=True, figsize=(12,12))
    fig.suptitle('주간 주성분 계수에 따른 3가지 표현 (80개의 클러스터를 다시 저차원의 형태로 변환)', fontsize=25)


  

    for PC_coeff in range(numComponents):
        meanTrace = weeklyTraces[:,PC_coeff,:].mean(axis=1)
        axArray[PC_coeff].plot(weeklyTraces[:,PC_coeff,:],'y',linewidth=1.5)
        axArray[PC_coeff].plot(meanTrace,'k',linewidth=2.5)
        axArray[PC_coeff].set_ylabel('PC %d coeff' %(PC_coeff+1))
        axArray[PC_coeff].vlines([0,23,47,71,95,119,143,167], weeklyTraces[:,PC_coeff,:].min(), weeklyTraces[:,PC_coeff,:].max(), colors='r', lw=2)

    axArray[PC_coeff].set_xlabel('주중 시간 경과에따른 변화')
    axArray[PC_coeff].set_xlim(-0.9,24*7-0.1)

    plt.savefig('{}/{}'.format(path,'pca_graph'))
    plt.clf()
    countFile('pca_graph')
        # collect traces for weekdays and weekends 
    listOfFullWeekdays = []
    listOfFullWeekends = []
    for uniqueVal in np.unique(dayOfYearVec):
        if (dayOfYearVec == uniqueVal).sum() == 24:
            if weekdayVec[dayOfYearVec == uniqueVal][0] <= 4:
                listOfFullWeekdays.append(uniqueVal)
            else:
                listOfFullWeekends.append(uniqueVal)

    weekdayTraces = np.zeros((24,numComponents,len(listOfFullWeekdays)))
    for k, dayInd in enumerate(listOfFullWeekdays):
        weekdayTraces[:,:,k] = compactClusterDistributionHourOfYear[dayOfYearVec == dayInd,:]

    weekendTraces = np.zeros((24,numComponents,len(listOfFullWeekends)))
    for k, dayInd in enumerate(listOfFullWeekends):
        weekendTraces[:,:,k] = compactClusterDistributionHourOfYear[dayOfYearVec == dayInd,:]

    fig, axArray = plt.subplots(nrows=numComponents,ncols=2,sharex=True,sharey=True, figsize=(12,14))
    fig.suptitle('주중 주말 사이의 주성분계수', fontsize=25)
   
    for PC_coeff in range(numComponents):
        axArray[PC_coeff][0].plot(weekdayTraces[:,PC_coeff,:],'c',linewidth=1.5)
        axArray[PC_coeff][0].plot(weekdayTraces[:,PC_coeff,:].mean(axis=1),'k',linewidth=2.5)
        axArray[PC_coeff][0].set_ylabel('PC %d coeff' %(PC_coeff+1))
        
        axArray[PC_coeff][1].plot(weekendTraces[:,PC_coeff,:],'c',linewidth=1.5)
        axArray[PC_coeff][1].plot(weekendTraces[:,PC_coeff,:].mean(axis=1),'k',linewidth=2.5)
        
        if PC_coeff == 0:
            axArray[PC_coeff][0].set_title('Weekday')
            axArray[PC_coeff][1].set_title('Weekend')
        
    axArray[PC_coeff][0].set_xlabel('hours of day')
    axArray[PC_coeff][1].set_xlabel('hours of day')
    axArray[PC_coeff][0].set_xlim(0,23)
    axArray[PC_coeff][0].set_ylim(-3.5,3.5)

    plt.savefig('{}/{}'.format(path,'weekday_weekend'))
    plt.clf()
    countFile('weekday_weekend')

    fig, axArray = plt.subplots(nrows=numComponents,ncols=1,sharex=True, figsize=(12,11))
    fig.suptitle('주성분 계수에 따른 운행 분포', fontsize=25)
    for PC_coeff in range(numComponents):
        tripTemplateDistributionDifference = TripDistributionPCAModel.components_[PC_coeff,:] * \
                                            TripDistributionPCAModel.explained_variance_[PC_coeff]
        axArray[PC_coeff].bar(range(1,numClusters+1),tripTemplateDistributionDifference)
        axArray[PC_coeff].set_title('주성분계수 %d ' %(PC_coeff+1))
        axArray[PC_coeff].set_ylabel('빈도수 [개수]')
        
    axArray[PC_coeff].set_xlabel('클러스터 인덱스 (sorted by cluster frequency)')
    axArray[PC_coeff].set_xlim(0,numClusters+0.5)

    axArray[1].hlines([-25,25], 0, numClusters+0.5, colors='r', lw=0.7)
    axArray[2].hlines([-11,11], 0, numClusters+0.5, colors='r', lw=0.7)

    plt.savefig('{}/{}'.format(path,'trip_distribution_of_PCA'))
    plt.clf()
    countFile('trip_distribution_of_PCA')

    numTopTripsToShow = 8
    numBottomTripsToShow = 6

    # meaning of 2nd PC
    sortedTripClusters_PC2 = np.argsort(TripDistributionPCAModel.components_[1,:])
    topPositiveTripClusterInds = sortedTripClusters_PC2[-numTopTripsToShow:]
    topNegativeTripClusterInds = sortedTripClusters_PC2[:numBottomTripsToShow]
    allInds = np.hstack((topPositiveTripClusterInds,topNegativeTripClusterInds))

    plt.figure(figsize=(12,12))
    plt.imshow(np.log(locationDensityImage+1),cmap='hot'); plt.grid('off')
    plt.scatter(srcImCoords[1][allInds],srcImCoords[0][allInds],c='m',s=500,alpha=0.9)
    plt.scatter(dstImCoords[1][allInds],dstImCoords[0][allInds],c='g',s=500,alpha=0.9)

    for i in topPositiveTripClusterInds:
        plt.arrow(srcImCoords[1][i],srcImCoords[0][i], dstImCoords[1][i]-srcImCoords[1][i], dstImCoords[0][i]-srcImCoords[0][i], 
                edgecolor='r', facecolor='r', width=2.8,alpha=0.9,head_width=10.0,head_length=10.0,length_includes_head=True)

    for i in topNegativeTripClusterInds:
        plt.arrow(srcImCoords[1][i],srcImCoords[0][i], dstImCoords[1][i]-srcImCoords[1][i], dstImCoords[0][i]-srcImCoords[0][i], 
                edgecolor='b', facecolor='b', width=2.8,alpha=0.9,head_width=10.0,head_length=10.0,length_includes_head=True)
    plt.title('주성분 계수2의 운행 기록')
    plt.savefig('{}/{}'.format(path,'trip_distribution_of_PCA2'))
    plt.clf()
    countFile('trip_distribution_of_PCA2')
        # meaning of 3rd PC
    numTopTripsToShow = 4
    numBottomTripsToShow = 10

    sortedTripClusters_PC3 = np.argsort(TripDistributionPCAModel.components_[2,:])
    topPositiveTripClusterInds = sortedTripClusters_PC3[-numTopTripsToShow:]
    topNegativeTripClusterInds = sortedTripClusters_PC3[:numBottomTripsToShow]
    allInds = np.hstack((topPositiveTripClusterInds,topNegativeTripClusterInds))

    plt.figure(figsize=(12,12))
    plt.imshow(np.log(locationDensityImage+1),cmap='hot'); plt.grid('off')
    plt.scatter(srcImCoords[1][allInds],srcImCoords[0][allInds],c='m',s=500,alpha=0.9)
    plt.scatter(dstImCoords[1][allInds],dstImCoords[0][allInds],c='g',s=500,alpha=0.9)

    for i in topPositiveTripClusterInds:
        plt.arrow(srcImCoords[1][i],srcImCoords[0][i], dstImCoords[1][i]-srcImCoords[1][i], dstImCoords[0][i]-srcImCoords[0][i], 
                edgecolor='r', facecolor='r', width=2.8,alpha=0.9,head_width=10.0,head_length=10.0,length_includes_head=True)

    for i in topNegativeTripClusterInds:
        plt.arrow(srcImCoords[1][i],srcImCoords[0][i], dstImCoords[1][i]-srcImCoords[1][i], dstImCoords[0][i]-srcImCoords[0][i], 
                edgecolor='b', facecolor='b', width=2.8,alpha=0.9,head_width=10.0,head_length=10.0,length_includes_head=True)
    plt.savefig('{}/{}'.format(path,'trip_distribution_of_PCA3'))
    plt.clf()
    countFile('trip_distribution_of_PCA3')

    with open(os.path.join(path,'{}.json'.format("extract")),'w',encoding='utf-8') as outfile:
        json.dump(output,outfile,ensure_ascii=False,indent='\t',cls=MyEncoder)

          

if __name__ == "__main__":
    
    # file_list = next(os.walk(ROOT_DIR))[2]
    file = sys.argv[1]
    # for i in file_list:
    #     if i.endswith('.csv'):
    #         file = i
    save_dir = os.path.join(ROOT_DIR,file.split('.')[0])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    run(file,save_dir)
 