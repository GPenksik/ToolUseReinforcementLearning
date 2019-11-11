clear
timestepStart = 1;
timestepLimit = 1000;
avgStep = 100; completeThreshold = 0.9;
RL = [2] ; RndL = [3]; Tsk = ["P" "L"];
runNumbers = [5010:5020];
runNumbers = [5021:5031];
runNumbers = [5041:5051]
seeds = [20:23];
nSeeds = length(seeds);

%runNumbers = [11:20];
%runNumbers = [201:209 273:275 210:218 276:278 ...
%               219:227 279:281 228:236 282:284 ...
%               237:245 285:287 246:254 288:290 ...
%               255:263 291:293 264:272 294:296];
for m = 1:length(Tsk)
    Tskx = Tsk(m);
    for n = 1:length(runNumbers)
        number = num2str(runNumbers(n),'%03.f');%(num,'%03.f')
        %for u = 1:nParams
            for u=1:length(seeds)
                RLx = num2str(RL(1));
                RndLx = num2str(RndL(1));
                number = num2str(runNumbers(n),'%03.f');%(num,'%03.f')
                seed = num2str(seeds(u), '%2.f');
                nameTemp = (dir (number + "-" + Tskx + RLx + RndLx + "*" + seed + "*.csv"));
                names(m,n,u) = string(nameTemp.name);
            end
        %end
    end
end
%
nRuns = length(names);
maxReward = 6.0
completedThreshold = completeThreshold*maxReward;
Data = cell([length(Tsk),nRuns,nSeeds]);
meanData  = cell([length(Tsk),nRuns,nSeeds]);
meanCompleted  = cell([length(Tsk),nRuns,nSeeds]);
completed = cell([length(Tsk),nRuns,nSeeds]);

for j=1:length(Tsk)
    for i=1:length(runNumbers)
        for k=1:length(seeds)
            if names(j,i,k) == ""
                Data{j,i,k} = zeros([timestepLimit - timestepStart + 1,1]);
            else
                Data{j,i,k} = ImportCSV(names(j,i,k), 1);
            end
            Data{j,i,k} = Data{j,i,k}(timestepStart:end);
            if (length(Data{j,i,k})>timestepLimit)
                Data{j,i,k} = Data{j,i,k}(1:timestepLimit);
            end
            meanData{j,i,k} = movmean(Data{j,i,k},avgStep);
            completed{j,i,k} = (Data{j,i,k}>(completedThreshold));
            meanCompleted{j,i,k} = movmean(completed{j,i,k},avgStep).*100;
        end
    end
end
%
figure(2)
clf
hold on
j = 2;
nSeeds = length(seeds);
for i=1:nRuns
    plotNum = mod(i,nRuns);
    if plotNum == 0 plotNum = nRuns;, end;
    subplot(3,4,plotNum);
    hold on
    for k=1:nSeeds
        plot(meanCompleted{j,i,k},'DisplayName',names(j,i,k))
    end
    ylim([0 100])
    xlim([0 timestepLimit-timestepStart])
end
%legend
%
figure(1)
clf
hold on
for i=1:nRuns
    plotNum = mod(i,nRuns);
    if plotNum == 0 plotNum = nRuns;, end;
    subplot(3,4,plotNum);
    hold on
    for k=1:nSeeds
        plot(meanData{j,i,k},'DisplayName',names(j,i,k))
    end
    ylim([0 6])
    xlim([0 timestepLimit-timestepStart])
end
%legend

%%
nParams = 9;
for j = 1:length(Tsk)
    for i = 1:nParams
        for k = 1:nRuns
            normData{j,k,i} = Data{j,k,i}/completedThreshold;
            normMeanData{j,k,i} = movmean(normData{j,k,i},avgStep);

            AvgData{j,k,i} = mean(normData{j,k,i});
            MaxData{j,k,i} = max(normMeanData{j,k,i});
            MaxCompeted{j,k,i} = max(meanCompleted{j,k,i});
            TotalCompleted{j,k,i} = mean(Data{j,k,i}>(completedThreshold)).*100;
        end
    end
end
%
% AvgDataC = mean(reshape(AvgData,nParams,[]),2);
% MaxDataC = mean(reshape(MaxData,nParams,[]),2);
% MaxCompetedC = mean(reshape(MaxCompeted,nParams,[]),2);
% TotalCompletedC = mean(reshape(TotalCompleted,nParams,[]),2);
% AvgDataCstd = std(reshape(AvgData,nParams,[]),0,2);
% MaxDataCstd = std(reshape(MaxData,nParams,[]),0,2);
% MaxCompetedCstd = std(reshape(MaxCompeted,nParams,[]),0,2);
% TotalCompletedCstd = std(reshape(TotalCompleted,nParams,[]),0,2);
% %%
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
%%
j=1;
DataSlice = reshape(Data(j,:,:),1,[]);
plotArray = cell2mat(DataSlice);
%plotArray = movmean(cell2mat(Data),100);
figure(7)
clf
grouping = [];
for i=1:9
    grouping = [grouping repmat(i,[1,10])]
end
boxplot(plotArray, grouping,'whisker',20)
