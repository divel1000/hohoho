from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt

now=(int(input("현재 진행중:1, 끝난 게임: 2")))
df1=pd.read_csv("10min_ratio.csv",encoding='cp949')
df2=pd.read_csv("Challenger_Ranked_Games.csv",encoding='cp949')
if now==1:
    team = (int(input("블루팀: 1, 레드팀: 2")))
    if team==1:
        features = df1[['blueFirstBlood', 'blueKills', 'blueDeaths', 'blueDragons', 'blueHeralds', 'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled']]
        win = df1['blueWins']
        train_features, test_features, train_labels, test_labels = train_test_split(features, win, train_size=(1500/2000),test_size=(500/2000), shuffle=False)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        model = LogisticRegression()
        model.fit(train_features, train_labels)
        values = list(map(int,input("퍼블(성공:1, 실패:0), 킬수, 데스수, 용(1 or 0), 전령(1 or 0), 총cs수, 정글 cs수를 입력하시오").split()))
        sample_values = np.array([values])
        sample_values = scaler.transform(sample_values)
        if model.predict(sample_values)[0]==1:
            print(model.predict_proba(sample_values)[0][1], " 의 확률로 이길확률이 더 높습니다.")
        else:
            print(model.predict_proba(sample_values)[0][1]," 의 확률로 질 확률이 더 높습니다.")
    elif team==2:
        features = df1[['redFirstBlood', 'redKills', 'redDeaths', 'redDragons', 'redHeralds', 'redTotalMinionsKilled','redTotalJungleMinionsKilled']]
        win = df1['redWins']
        train_features, test_features, train_labels, test_labels = train_test_split(features, win, train_size=(1500/2000),test_size=(500/2000), shuffle=False)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        model = LogisticRegression()
        model.fit(train_features, train_labels)
        values = list(map(int, input("퍼블(성공:1, 실패:0), 킬수, 데스수, 용(1 or 0), 전령(1 or 0),  총cs수, 정글 cs수를 입력하시오").split()))
        sample_values = np.array([values])
        sample_values = scaler.transform(sample_values)
        if model.predict(sample_values)[0] == 1:
            print(model.predict_proba(sample_values)[0][1], " 의 확률로 이길확률이 더 높습니다.")
        else:
            print(model.predict_proba(sample_values)[0][1], " 의 확률로 질 확률이 더 높습니다.")

#종료된 게임은 시간별 평균 데이터를 가지고 가장 최근의 게임 데이터와 비교한다.
elif now==2:
    user_name=input("소환사 아이디를 입력하세요: ")
    #라이엇 api 허가된 시간 이후에는 갱신 필요
    api_key = 'RGAPI-cee40ba0-5b31-45ae-92da-4b48d983af85'
    #원하는 아이디의 정보를 불러옴
    sohwan = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/"+ user_name + "?api_key=" + api_key
    r = requests.get(sohwan)

    match_info_df=pd.DataFrame()
    season = str(13)
    matches_url="https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/" + r.json()['accountId'] + "?api_key=" + api_key
    r2=requests.get(matches_url)
    match_info_df=pd.concat([match_info_df, pd.DataFrame(r2.json()['matches'])])

    match_info_df2=match_info_df
    match=pd.DataFrame()
    api_url = 'https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_info_df2['gameId'].iloc[0]) + '?api_key=' + api_key
    r5 = requests.get(api_url)

    # 위의 예외처리 코드를 거쳐서 내려왔을 때 해당 코드가 실행될 수 있도록 작성
    mat = pd.DataFrame(list(r5.json().values()), index=list(r5.json().keys())).T
    match_fin=pd.DataFrame()
    match_fin = pd.concat([match_fin, mat])

    #매치정보에서 필요한 정보만을 추출하는 단계
    a_ls=list(r5.json()['teams'])
    a=r5.json()['gameDuration']

    #각 팀별 킬,데스 어시스트를 구하는 단계
    red_kda=pd.DataFrame()
    blue_kda=pd.DataFrame()
    red_kills=0
    red_deaths=0
    red_assists=0
    red_cs=0
    blue_kills = 0
    blue_deaths = 0
    blue_assists = 0
    blue_cs=0
    for i in range(0,5):
        red_kills += r5.json()['participants'][i]['stats']['kills']
        red_deaths += r5.json()['participants'][i]['stats']['deaths']
        red_assists += r5.json()['participants'][i]['stats']['assists']
        red_cs += r5.json()['participants'][i]['stats']['totalMinionsKilled']
    for i in range(5,10):
        blue_kills += r5.json()['participants'][i]['stats']['kills']
        blue_deaths += r5.json()['participants'][i]['stats']['deaths']
        blue_assists += r5.json()['participants'][i]['stats']['assists']
        blue_cs += r5.json()['participants'][i]['stats']['totalMinionsKilled']

    # 사용자가 어떤팀에 속해있는지 파악한후 데이터 수집
    for i in range(10):
        if r.json()['id']==r5.json()['participantIdentities'][i]['player']['summonerId']:
            #i값이 5이상으면 레드팀, 5미만이면 블루팀으로 구분하여 실행
            if i>=5:
                team_color=200
                team2_df = pd.DataFrame()
                for i in range(1):
                    try:
                        a_ls[1].pop('bans')
                        team2 = pd.DataFrame(list(a_ls[1].values()), index=list(a_ls[1].keys())).T
                        team2_df = team2_df.append(team2)
                    except:
                        pass
                team2_df.index = range(len(team2_df))
                team2_df['Kills']=red_kills
                team2_df['Deaths']=red_deaths
                team2_df['Assists']=red_assists
                team2_df['totalMinionsKilled']=red_cs
                team2_df['gameDuration']=a
                del team2_df['vilemawKills']
                del team2_df['firstRiftHerald']
                del team2_df['dominionVictoryScore']
                del team2_df['riftHeraldKills']

                #문자열로 구성되었던 컬럼을 1과0으로 인덱싱
                team2_df.loc[team2_df.win=='Win', 'win']=int(1)
                team2_df.loc[team2_df.win == 'Fail', 'win'] = int(0)
                team2_df.loc[team2_df.firstBlood == True, 'firstBlood'] = int(1)
                team2_df.loc[team2_df.firstBlood == False, 'firstBlood'] = int(0)
                team2_df.loc[team2_df.firstTower ==True, 'firstTower'] = int(1)
                team2_df.loc[team2_df.firstTower == False, 'firstTower'] = int(0)
                team2_df.loc[team2_df.firstInhibitor == True, 'firstInhibitor'] = int(1)
                team2_df.loc[team2_df.firstInhibitor == False, 'firstInhibitor'] = int(0)
                team2_df.loc[team2_df.firstBaron == True, 'firstBaron'] = int(1)
                team2_df.loc[team2_df.firstBaron == False, 'firstBaron'] = int(0)
                team2_df.loc[team2_df.firstDragon == True, 'firstDragon'] = int(1)
                team2_df.loc[team2_df.firstDragon == False, 'firstDragon'] = int(0)

                df3=df2[['gameDuration', 'redFirstBlood', 'redFirstTower', 'redFirstInhibitor', 'redFirstBaron',
                         'redFirstDragon', 'redTowerKills', 'redInhibitorKills', 'redBaronKills', 'redDragonKills',
                         'redKills', 'redDeath', 'redAssist', 'redTotalMinionKills']]

                #게임시간이 10분을 넘기지 않을경우 실행하지 않고 넘을경우 3분간격의 평균데이터와 비교한다.
                gametime=int(team2_df['gameDuration']//60)
                if gametime<10:
                    print('게임진행 시간이 너무 짧습니다.')
                else:
                    for i in range(len(df3)):
                        if int(df3.iloc[i]['gameDuration'])>=gametime:
                            break;
                    resultdf=({'firstBlood':[float(df3.iloc[i]['redFirstBlood'])-float(team2_df['firstBlood'])],
                           'firstTower':[float(df3.iloc[i]['redFirstTower'])-float(team2_df['firstTower'])],
                           'firstInhibitor':[float(df3.iloc[i]['redFirstInhibitor'])-float(team2_df['firstInhibitor'])],
                           'firstBaron':[float(df3.iloc[i]['redFirstBaron'])-float(team2_df['firstBaron'])],
                           'firstDragon':[float(df3.iloc[i]['redFirstDragon'])-float(team2_df['firstDragon'])],
                           'towerKills':[float(df3.iloc[i]['redTowerKills'])-float(team2_df['towerKills'])],
                           'inhibitorKills':[float(df3.iloc[i]['redInhibitorKills'])-float(team2_df['inhibitorKills'])],
                           'baronKills':[float(df3.iloc[i]['redBaronKills'])-float(team2_df['baronKills'])],
                           'dragonKills':[float(df3.iloc[i]['redDragonKills'])-float(team2_df['dragonKills'])],
                           'Kills':[float(df3.iloc[i]['redKills'])-float(team2_df['Kills'])],
                           'Deaths':[float(df3.iloc[i]['redDeath'])-float(team2_df['Deaths'])],
                           'Assists':[float(df3.iloc[i]['redAssist'])-float(team2_df['Assists'])],
                           'MinionsKill':[float(df3.iloc[i]['redTotalMinionKills'])-float(team2_df['totalMinionsKilled'])]})
                    # 비교하는데 필요없는 게임시간 삭제
                    del team2_df['gameDuration']

                    # 상위 티어와 비교한 값을 시각화 하는 과정
                    resultdf = pd.DataFrame(resultdf)
                    resultdf = resultdf.T
                    print(team2_df)
                    print(resultdf)

                    ax = resultdf.plot(kind='bar', rot=0, figsize=(15, 5))
                    plt.title("Challenger Average-My Data")
                    plt.xlabel(user_name+" Data")
                    plt.ylabel("Comparison value")
                    for p in ax.patches:
                        left, bottom, width, height = p.get_bbox().bounds
                        ax.annotate("%.3f" % (height), (left + width / 2, height * 1.01), ha='center')

                    plt.show()
            else:
                team_color=100
                team1_df = pd.DataFrame()
                for i in range(1):
                    try:
                        a_ls[0].pop('bans')
                        team1 = pd.DataFrame(list(a_ls[0].values()), index=list(a_ls[0].keys())).T
                        team1_df = team1_df.append(team1)
                    except:
                        pass
                team1_df.index = range(len(team1_df))
                team1_df['Kills'] = blue_kills
                team1_df['Deaths'] = blue_deaths
                team1_df['Assists'] = blue_assists
                team1_df['totalMinionsKilled'] = blue_cs
                team1_df['gameDuration'] = a
                del team1_df['vilemawKills']
                del team1_df['firstRiftHerald']
                del team1_df['dominionVictoryScore']
                del team1_df['riftHeraldKills']

                # 문자열로 구성되었던 컬럼을 1과0으로 인덱싱
                team1_df.loc[team1_df.win == 'Win', 'win'] = int(1)
                team1_df.loc[team1_df.win == 'Fail', 'win'] = int(0)
                team1_df.loc[team1_df.firstBlood == True, 'firstBlood'] = int(1)
                team1_df.loc[team1_df.firstBlood == False, 'firstBlood'] = int(0)
                team1_df.loc[team1_df.firstTower == True, 'firstTower'] = int(1)
                team1_df.loc[team1_df.firstTower == False, 'firstTower'] = int(0)
                team1_df.loc[team1_df.firstInhibitor == True, 'firstInhibitor'] = int(1)
                team1_df.loc[team1_df.firstInhibitor == False, 'firstInhibitor'] = int(0)
                team1_df.loc[team1_df.firstBaron == True, 'firstBaron'] = int(1)
                team1_df.loc[team1_df.firstBaron == False, 'firstBaron'] = int(0)
                team1_df.loc[team1_df.firstDragon == True, 'firstDragon'] = int(1)
                team1_df.loc[team1_df.firstDragon == False, 'firstDragon'] = int(0)

                gametime = int(team1_df['gameDuration'] // 60)

                #사용자의 모든 데이터에서 필요한 데이터만을 구분하여 추출
                df3 = df2[['gameDuration', 'blueFirstBlood', 'blueFirstTower', 'blueFirstInhibitor', 'blueFirstBaron',
                           'blueFirstDragon', 'blueTowerKills', 'blueInhibitorKills', 'blueBaronKills', 'blueDragonKills',
                           'blueKills', 'blueDeath', 'blueAssist', 'blueTotalMinionKills']]

                # 게임시간이 10분을 넘기지 않을경우 실행하지 않고 넘을경우 3분간격의 평균데이터와 비교한다.
                gametime = int(team1_df['gameDuration'] // 60)
                if gametime < 10:
                    print('게임진행 시간이 너무 짧습니다.')
                else:
                    for i in range(len(df3)):
                        if int(df3.iloc[i]['gameDuration']) >= gametime:
                            break;

                    resultdf = ({'firstBlood': [float(df3.iloc[i]['blueFirstBlood']) - float(team1_df['firstBlood'])],
                                 'firstTower': [float(df3.iloc[i]['blueFirstTower']) - float(team1_df['firstTower'])],
                                 'firstInhibitor': [float(df3.iloc[i]['blueFirstInhibitor']) - float(team1_df['firstInhibitor'])],
                                 'firstBaron': [float(df3.iloc[i]['blueFirstBaron']) - float(team1_df['firstBaron'])],
                                 'firstDragon': [float(df3.iloc[i]['blueFirstDragon']) - float(team1_df['firstDragon'])],
                                 'towerKills': [float(df3.iloc[i]['blueTowerKills']) - float(team1_df['towerKills'])],
                                 'inhibitorKills': [float(df3.iloc[i]['blueInhibitorKills']) - float(team1_df['inhibitorKills'])],
                                 'baronKills': [float(df3.iloc[i]['blueBaronKills']) - float(team1_df['baronKills'])],
                                 'dragonKills': [float(df3.iloc[i]['blueDragonKills']) - float(team1_df['dragonKills'])],
                                 'Kills': [float(df3.iloc[i]['blueKills']) - float(team1_df['Kills'])],
                                 'Deaths': [float(df3.iloc[i]['blueDeath']) - float(team1_df['Deaths'])],
                                 'Assists': [float(df3.iloc[i]['blueAssist']) - float(team1_df['Assists'])],
                                 'MinionsKill': [float(df3.iloc[i]['blueTotalMinionKills']) - float(team1_df['totalMinionsKilled'])]})

                    #비교하는데 필요없는 게임시간 삭제
                    del team1_df['gameDuration']

                    #상위 티어와 비교한 값을 시각화 하는 과정
                    resultdf = pd.DataFrame(resultdf)
                    resultdf=resultdf.T
                    print(team1_df)
                    print(resultdf)
                    type(resultdf)

                    ax = resultdf.plot(kind='bar', rot=0,figsize=(15,5))
                    plt.title("Challenger Average-My Data")
                    plt.xlabel(user_name+" Data")
                    plt.ylabel("Comparison value")
                    for p in ax.patches:
                        left, bottom, width, height = p.get_bbox().bounds
                        ax.annotate("%.1f" % (height), (left + width / 2, height * 1.01), ha='center')

                    plt.show()

else:
    print('프로그램을 종료합니다.')