clear
timestepStart = 1;
timestepLimit = 1000;
avgStep = 100; completeThreshold = 0.9;
RL = 3; Tsk = "P"
runNumbers = [301:309 373:375 310:318 376:378 ...
              319:327 379:381 328:336 382:384 ...
              337:345 385:387 346:354 388:390 ...
              355:363 391:393 364:372 394:396];
% runNumbers = [201:209 273:275 210:218 276:278 ...
%               219:227 279:281 228:236 282:284 ...
%               237:245 285:287 246:254 288:290 ...
%               255:263 291:293 264:272 294:296];
for n = 1:length(runNumbers)
    number = num2str(runNumbers(n));
    nameTemp = (dir (number + "*" + Tsk + RL + "3*.csv"));
    names(n) = string(nameTemp.name);
end
%
nRuns = length(names);
for o=1:nRuns
    nLevel = mod(o,3);
    if nLevel==1
        if RL == 2
            completedThreshold(o) = completeThreshold*2.0;
        elseif RL == 3
            completedThreshold(o) = 1+completeThreshold*1.0;
        end
    elseif nLevel == 2
        if RL == 2
            completedThreshold(o) = completeThreshold*6.0;
        elseif RL == 3
            completedThreshold(o) = 1 + completeThreshold*5.0;
        end
    else
        if RL == 2
            completedThreshold(o) = completeThreshold*60;
        elseif RL == 3
            completedThreshold(o) = 1 + completeThreshold*59.0;
        end
        
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
    plotNum = mod(j,12);
    if plotNum == 0 plotNum = 12;, end;
    subplot(4,3,plotNum);
    hold on
    plot(meanCompleted{j},'DisplayName',names(j))
    ylim([0 100])
    xlim([0 timestepLimit-timestepStart])
end
%legend

figure(1)
clf

for k=index
    plotNum = mod(k,12);
    if plotNum == 0 plotNum = 12;, end;
    subplot(4,3,plotNum);
    hold on
    plot(meanData{k},'DisplayName',names(k))
    if RL == 3
        ylim([0 6]) %completedThreshold(k)])
    elseif RL == 2
        ylim([-1 completedThreshold(k)])
    end
    xlim([0 timestepLimit-timestepStart])
end
%legend
%%
figure(4)
clf
hold on
j=1
indexes = [2, 5, 8, 11];
nSeeds = 6;
meanData2 = reshape(meanData, [1 12 8])
names2 = reshape(names, [1 12 8])
for i=1:4
    subplot(1,4,i);
    hold on
    i2 = indexes(i);
    for k=1:nSeeds
        plot(meanData2{j,i2,k})
    end
    ylim([0 6])
    xlim([0 timestepLimit-timestepStart])
    set(gca,'XTick',[0 250 500 750 999],'XTickLabel',{'0','','','','1000'})
    grid
    box on
    if i == 1
        title("Number of repeats = 3")
        xlabel("Episode")
        ylabel("Moving average of reward")
    elseif i == 2
        title("Number of repeats = 6")
        xlabel("Episode")
        ylabel("Moving average of reward")
    elseif i == 3
        title("Number of repeats = 12")
        xlabel("Episode")
        ylabel("Moving average of reward")
    elseif i == 4
        title("Number of repeats = 16")
        xlabel("Episode")
        ylabel("Moving average of reward")
    end
end

% %%
% nParams = 12;
% 
% for l = 1:nRuns
%     normData{l} = Data{l}/completedThreshold(l);
%     normMeanData{l} = movmean(normData{l},avgStep);
% 
%     AvgData(l) = mean(normData{l});
%     MaxData(l) = max(normMeanData{l});
%     MaxCompeted(l) = max(meanCompleted{l});
%     TotalCompleted(l) = mean(Data{l}>(completedThreshold(l))).*100;
% end
% 
% AvgDataC = mean(reshape(AvgData,nParams,[]),2);
% MaxDataC = mean(reshape(MaxData,nParams,[]),2);
% MaxCompetedC = mean(reshape(MaxCompeted,nParams,[]),2);
% TotalCompletedC = mean(reshape(TotalCompleted,nParams,[]),2);
% AvgDataCstd = std(reshape(AvgData,nParams,[]),0,2);
% MaxDataCstd = std(reshape(MaxData,nParams,[]),0,2);
% MaxCompetedCstd = std(reshape(MaxCompeted,nParams,[]),0,2);
% TotalCompletedCstd = std(reshape(TotalCompleted,nParams,[]),0,2);
% 
% figure(3)
% clf
% offset = 7;
% index3 = [7;2;6];
% index2 = [0+offset;3+offset;6+offset;9+offset];
% index1 = [0+offset;1+offset;2+offset];
% index = index1;
% plot(AvgDataC(index)/max(AvgDataC))
% hold on
% plot(MaxDataC(index)/max(MaxDataC))
% plot(MaxCompetedC(index)/max(MaxCompetedC))
% plot(TotalCompletedC(index)/max(TotalCompletedC))
% ylim([0 1])
% legend
% figure(4)
% clf
% plot(AvgDataCstd(index)/max(AvgDataCstd))
% hold on
% plot(MaxDataCstd(index)/max(MaxDataCstd))
% plot(MaxCompetedCstd(index)/max(MaxCompetedCstd))
% plot(TotalCompletedCstd(index)/max(TotalCompletedCstd))
% ylim([0 1])
% legend
% %%
% figure(5)
% clf
% plotArray = AvgData;
% bar(reshape(plotArray,nParams,[]))

