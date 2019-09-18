clear
timestepLimit = 2000;
avgStep = 300; names = [];
names = [names; "901-P33-20-PPO.csv"];
names = [names; "902-P33-20-PPO.csv"];
names = [names; "903-P33-20-PPO.csv"];
names = [names; "904-P33-20-PPO.csv"];
names = [names; "905-P33-20-PPO.csv"];
names = [names; "906-P33-20-PPO.csv"];
names = [names; "907-P33-20-PPO.csv"];
nRuns = length(names);
completedThreshold = ones([length(names),1])*5.5;
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
    MaxCompeted(l) = max(meanCompleted{l})
    TotalCompleted(l) = mean(Data{l}>1).*100;
end


