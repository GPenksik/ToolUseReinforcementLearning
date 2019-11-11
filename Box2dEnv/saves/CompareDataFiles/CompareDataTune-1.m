clear
timestepLimit = 2000;
avgStep = 200; names = []; completeThreshold = 5.5;
names = [names; "901-P33-20-PPO.csv"];
names = [names; "902-P33-20-PPO.csv"];
names = [names; "903-P33-20-PPO.csv"];
names = [names; "904-P33-20-PPO.csv"];
names = [names; "905-P33-20-PPO.csv"];
names = [names; "906-P33-20-PPO.csv"];
names = [names; "907-P33-20-PPO.csv"];
names = [names; "908-P33-21-PPO.csv"];
names = [names; "909-P33-21-PPO.csv"];
names = [names; "910-P33-21-PPO.csv"];
names = [names; "911-P33-21-PPO.csv"];
names = [names; "912-P33-21-PPO.csv"];
names = [names; "913-P33-21-PPO.csv"];
names = [names; "914-P33-21-PPO.csv"];
names = [names; "915-P33-22-PPO.csv"];
names = [names; "916-P33-22-PPO.csv"];
names = [names; "917-P33-22-PPO.csv"];
names = [names; "918-P33-22-PPO.csv"];
names = [names; "919-P33-22-PPO.csv"];
names = [names; "920-P33-22-PPO.csv"];
names = [names; "921-P33-22-PPO.csv"];
nRuns = length(names);
completedThreshold = ones([length(names),1])*completeThreshold;
for i=1:nRuns
    Data{i} = ImportCSV(names(i), 1);
    if (length(Data{i})>timestepLimit)
        Data{i} = Data{i}(1:timestepLimit);
    end
    meanData{i} = movmean(Data{i},avgStep);
    completed{i} = (Data{i}>completedThreshold(i));
    meanCompleted{i} = movmean(completed{i},avgStep).*100;
end
%
figure(2)
clf
hold on
for j=1:nRuns
    plot(meanCompleted{j},'DisplayName',names(j))
end
legend
ylim([0 100])
figure(1)
clf
hold on
for k=1:nRuns
    plot(meanData{k},'DisplayName',names(k))
end
legend

%%
for l = 1:nRuns
    AvgData(l) = mean(Data{l});
    MaxData(l) = max(meanData{l});
    MaxCompeted(l) = max(meanCompleted{l});
    TotalCompleted(l) = mean(Data{l}>completeThreshold).*100;
end
AvgDataC = mean(reshape(AvgData,[7,3]),2);
MaxDataC = mean(reshape(MaxData,[7,3]),2);
MaxCompetedC = mean(reshape(MaxCompeted,[7,3]),2);
TotalCompletedC = mean(reshape(TotalCompleted,[7,3]),2);

figure(3)
clf
index3 = [4;2;5];
index2 = [7;2;6];
index1 = [3;2;1]
index = index1;
plot(AvgDataC(index)/max(AvgDataC(index)))
hold on
plot(MaxDataC(index)/max(MaxDataC(index)))
plot(MaxCompetedC(index)/max(MaxCompetedC(index)))
plot(TotalCompletedC(index)/max(TotalCompletedC(index)))
ylim([0 1])
