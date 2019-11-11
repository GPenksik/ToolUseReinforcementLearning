clear
timestepLimit = 2000;
avgStep = 100; names = []; completeThreshold = 5.5;
names = [names; "902-P33-20-PPO.csv"]; 
names = [names; "751-P32-20-PPO.csv"];
names = [names; "754-P23-20-PPO.csv"];
names = [names; "757-P22-20-PPO.csv"];
names = [names; "909-P33-21-PPO.csv"];
names = [names; "752-P32-21-PPO.csv"];
names = [names; "755-P23-21-PPO.csv"];
names = [names; "758-P22-21-PPO.csv"];
names = [names; "916-P33-22-PPO.csv"];
names = [names; "753-P32-22-PPO.csv"];
names = [names; "756-P23-22-PPO.csv"];
names = [names; "759-P22-22-PPO.csv"];
nRuns = length(names);
for o=1:nRuns
    rewardlevel = str2num(extractBetween(names(o),6,6));
    if rewardlevel==3
        completedThreshold(o) = 5.5;
    elseif rewardlevel == 2
        completedThreshold(o) = 4.5;
    else
        completedThreshold(o) = 10;
    end
end
%
for i=1:nRuns
    Data{i} = ImportCSV(names(i), 1);
    if (length(Data{i})>timestepLimit)
        Data{i} = Data{i}(1:timestepLimit);
    end
    meanData{i} = movmean(Data{i},avgStep);
    completed{i} = (Data{i}>completedThreshold(i));
    meanCompleted{i} = movmean(completed{i},avgStep).*100;
end
%%
figure(2)
clf
hold on
offset = 4;
indexall = [1:12];
index1 = [0+offset,4+offset,8+offset];
index = index1;
for j=index
    plot(meanCompleted{j},'DisplayName',names(j))
end
legend
ylim([0 100])
figure(1)
clf
hold on
for k=index
    plot(meanData{k},'DisplayName',names(k))
end
ylim([-1 completeThreshold + 1])
legend

%%
for l = 1:nRuns
    AvgData(l) = mean(Data{l});
    MaxData(l) = max(meanData{l});
    MaxCompeted(l) = max(meanCompleted{l});
    TotalCompleted(l) = mean(Data{l}>completeThreshold).*100;
end
AvgDataC = mean(reshape(AvgData,[4,3]),2);
MaxDataC = mean(reshape(MaxData,[4,3]),2);
MaxCompetedC = mean(reshape(MaxCompeted,[4,3]),2);
TotalCompletedC = mean(reshape(TotalCompleted,[4,3]),2);

figure(3)
clf
index3 = [1;2;3;4];
index2 = [7;2;6];
index1 = [3;2;1];
index = index3;
plot(AvgDataC(index)/max(AvgDataC(index)))
hold on
plot(MaxDataC(index)/max(MaxDataC(index)))
plot(MaxCompetedC(index)/max(MaxCompetedC(index)))
plot(TotalCompletedC(index)/max(TotalCompletedC(index)))
ylim([0 1])
