clear
timestepLimit = 2000;
avgStep = 100; names = []; completeThreshold = 5.5;
names = [names; "780-P33-20-PPO.csv"];
names = [names; "781-P33-20-PPO.csv"];
names = [names; "782-P33-20-PPO.csv"];
names = [names; "783-P33-20-PPO.csv"];
names = [names; "784-P33-20-PPO.csv"];
names = [names; "785-P33-20-PPO.csv"];
names = [names; "786-P33-20-PPO.csv"];
names = [names; "787-P33-20-PPO.csv"];
names = [names; "788-P33-20-PPO.csv"];
names = [names; "789-P33-20-PPO.csv"];
names = [names; "790-P33-20-PPO.csv"];
names = [names; "791-P33-20-PPO.csv"];
names = [names; "792-P33-20-PPO.csv"];
names = [names; "793-P33-20-PPO.csv"];
names = [names; "794-P33-20-PPO.csv"];
names = [names; "795-P33-20-PPO.csv"];
names = [names; "796-P33-20-PPO.csv"];
names = [names; "797-P33-20-PPO.csv"];
names = [names; "798-P33-20-PPO.csv"];
names = [names; "799-P33-20-PPO.csv"];

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
%legend
ylim([0 100])
figure(1)
clf
hold on
for k=1:nRuns
    plot(meanData{k},'DisplayName',names(k))
end
%legend

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
