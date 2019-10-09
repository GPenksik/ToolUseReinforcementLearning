clear
timestepStart = 501;
timestepLimit = 2000;
avgStep = 100; completeThreshold = 0.9;
RL = 3; Tsk = "L";
runNumbers = [1:48 52:75 79:102];
folder = "Backup/";
%runNumbers = [201:209 273:275 210:218 276:278 ...
%               219:227 279:281 228:236 282:284 ...
%               237:245 285:287 246:254 288:290 ...
%               255:263 291:293 264:272 294:296];
for n = 1:length(runNumbers)
    number = num2str(runNumbers(n),'%03.f');%(num,'%03.f')
    nameTemp = (dir (folder + "*" + number + "-*" + Tsk + RL + "*.csv"));
    names(n) = string(nameTemp.name);
end
%%
nRuns = length(names);
for o=1:nRuns
    completedThreshold(o) = completeThreshold*6.0
end
%
for i=1:nRuns
    Data{i} = ImportCSV(folder + names(i), 1);
    Data{i} = Data{i}(timestepStart:end);
    if (length(Data{i})>timestepLimit)
        Data{i} = Data{i}(1:timestepLimit);
    end
    meanData{i} = movmean(Data{i},avgStep);
    completed{i} = (Data{i}>(completedThreshold(i)));
    meanCompleted{i} = movmean(completed{i},avgStep).*100;
end
%
figure(2)
clf
hold on
offset = 6;
indexall = 1:nRuns;
index1 = [0+offset,7+offset,14+offset,21+offset]%, 28+offset];
index = indexall;
for j=index
    plotNum = mod(j,24);
    if plotNum == 0 plotNum = 24;, end;
    subplot(6,4,plotNum);
    hold on
    plot(meanCompleted{j},'DisplayName',names(j))
    ylim([0 100])
    xlim([0 timestepLimit-timestepStart])
end
%legend

figure(1)
clf

for k=index
    plotNum = mod(k,24);
    if plotNum == 0 plotNum = 24;, end;
    subplot(6,4,plotNum);
    hold on
    plot(meanData{k},'DisplayName',names(k))
    if RL == 3
        ylim([1 completedThreshold(k)])
    elseif RL == 2
        ylim([-1 completedThreshold(k)])
    end
    xlim([0 timestepLimit-timestepStart])
end
%legend

%%
nParams = 24;

for l = 1:nRuns
    normData{l} = Data{l}/completedThreshold(l);
    normMeanData{l} = movmean(normData{l},avgStep);

    AvgData(l) = mean(normData{l});
    MaxData(l) = max(normMeanData{l});
    MaxCompeted(l) = max(meanCompleted{l});
    TotalCompleted(l) = mean(Data{l}>(completedThreshold(l))).*100;
end

AvgDataC = mean(reshape(AvgData,nParams,[]),2);
MaxDataC = mean(reshape(MaxData,nParams,[]),2);
MaxCompetedC = mean(reshape(MaxCompeted,nParams,[]),2);
TotalCompletedC = mean(reshape(TotalCompleted,nParams,[]),2);
AvgDataCstd = std(reshape(AvgData,nParams,[]),0,2);
MaxDataCstd = std(reshape(MaxData,nParams,[]),0,2);
MaxCompetedCstd = std(reshape(MaxCompeted,nParams,[]),0,2);
TotalCompletedCstd = std(reshape(TotalCompleted,nParams,[]),0,2);

figure(3)
clf
offset = 7;
index3 = [7;2;6];
index2 = [0+offset;3+offset;6+offset;9+offset];
index1 = [0+offset;1+offset;2+offset];
index = index1;
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
plotArray = movmean(cell2mat(Data)>5.4,100);
%plotArray = movmean(cell2mat(Data),100);
for g=1:96
   if length(Data{g})~=1500
       g
   end
end
figure(7)
clf
grouping = repmat([1:24],[1,4])
boxplot(plotArray,grouping, 'whisker',5)
