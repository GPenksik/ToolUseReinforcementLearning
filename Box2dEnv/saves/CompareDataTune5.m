clear
timestepStart = 1;
timestepLimit = 2000;
avgStep = 200; completeThreshold = 5.5;
runNumbers = [601:635];
%runNumbers = [636:670];
for n = 1:length(runNumbers)
    number = num2str(runNumbers(n));
    nameTemp = (dir (number + "*.csv"));
    names(n) = string(nameTemp.name);
end

nRuns = length(names);
for o=1:nRuns
    rewardlevel = str2num(extractBetween(names(o),6,6));
    if rewardlevel==3
        completedThreshold(o) = 5.5;
    elseif rewardlevel == 2
        completedThreshold(o) = 5.5;
    else
        completedThreshold(o) = 10;
    end
end
%
for i=1:nRuns
    Data{i} = ImportCSV(names(i), 1);
    Data{i} = Data{i}(timestepStart:end)
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
offset = 7;
indexall = [1:nRuns];
index1 = [0+offset,7+offset,14+offset,21+offset, 28+offset];
index = index1;
for j=index
    plot(meanCompleted{j},'DisplayName',names(j))
end
%legend
ylim([0 100])
figure(1)
clf
hold on
for k=index
    plot(meanData{k},'DisplayName',names(k))
end
ylim([-1 completeThreshold + 1])
%legend

%%
for l = 1:nRuns
    AvgData(l) = mean(Data{l});
    MaxData(l) = max(meanData{l});
    MaxCompeted(l) = max(meanCompleted{l});
    TotalCompleted(l) = mean(Data{l}>completeThreshold).*100;
end
AvgDataC = mean(reshape(AvgData,7,[]),2);
MaxDataC = mean(reshape(MaxData,7,[]),2);
MaxCompetedC = mean(reshape(MaxCompeted,7,[]),2);
TotalCompletedC = mean(reshape(TotalCompleted,7,[]),2);
AvgDataCstd = std(reshape(AvgData,7,[]),0,2);
MaxDataCstd = std(reshape(MaxData,7,[]),0,2);
MaxCompetedCstd = std(reshape(MaxCompeted,7,[]),0,2);
TotalCompletedCstd = std(reshape(TotalCompleted,7,[]),0,2);

figure(3) 
clf
index3 = [7;2;6];
index2 = [4;2;5];
index1 = [3;2;1];
index = index3;
plot(AvgDataC(index)/max(AvgDataC))
hold on
plot(MaxDataC(index)/max(MaxDataC))
plot(MaxCompetedC(index)/max(MaxCompetedC))
plot(TotalCompletedC(index)/max(TotalCompletedC))
ylim([0 1])
legend
figure(4) 
clf
plot(AvgDataCstd(index)/max(AvgDataCstd))
hold on
plot(MaxDataCstd(index)/max(MaxDataCstd))
plot(MaxCompetedCstd(index)/max(MaxCompetedCstd))
plot(TotalCompletedCstd(index)/max(TotalCompletedCstd))
ylim([0 1])
legend

%%
figure(4)
clf
plotArray = AvgData;
bar(reshape(plotArray,7,[]))

