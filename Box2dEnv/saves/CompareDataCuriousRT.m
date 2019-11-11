clear
timestepStart = 1;
timestepLimit = 500;
avgStep = 100; completeThreshold = 0.9;
RL = [3 3] ; RndL = [3 3]; Tsk = ["P" "L"];
%RL = [3 2] ; RndL = [3 3];
nParams = length(RL);
runNumbers = [1:44];
for m_Task = 1:length(Tsk)
    Tskx = Tsk(m_Task);
    for n_Run = 1:length(runNumbers)
        for u_Param = 1:nParams
            nameTempString = "";
            RLx = num2str(RL(u_Param));
            RndLx = num2str(RndL(u_Param));
            number = num2str(runNumbers(n_Run),'%03.f');%(num,'%03.f')
            nameTemp = (dir (number + "-*-RT" + ".csv"));
            nameTempString = string(nameTemp.name);
%             if nameTempString == ""
%                 nameTempString = "400-L33-20-PPO.csv";
%             end
            names(m_Task,n_Run,u_Param) = nameTempString;
        end
    end
end
%
nRuns = length(names);
rewardScale = 100.0;
completedThreshold = completeThreshold*rewardScale;
Data = cell([length(Tsk),nRuns,nParams]);
meanData  = cell([length(Tsk),nRuns,nParams]);
meanCompleted  = cell([length(Tsk),nRuns,nParams]);
completed = cell([length(Tsk),nRuns,nParams]);

for j_Task=1:length(Tsk)
    for i_Run=1:length(runNumbers)
        for k_Param=1:nParams
            if names(j_Task,i_Run,k_Param) == ""
                Data{j_Task,i_Run,k_Param} = zeros([timestepLimit - timestepStart + 1,1])
            else
                Data{j_Task,i_Run,k_Param} = ImportCSV(names(j_Task,i_Run,k_Param), 1);
            end
            Data{j_Task,i_Run,k_Param} = Data{j_Task,i_Run,k_Param}(timestepStart:end);
            if (length(Data{j_Task,i_Run,k_Param})>timestepLimit)
                Data{j_Task,i_Run,k_Param} = Data{j_Task,i_Run,k_Param}(1:timestepLimit);
            end
            meanData{j_Task,i_Run,k_Param} = movmean(Data{j_Task,i_Run,k_Param},avgStep);
            completed{j_Task,i_Run,k_Param} = (Data{j_Task,i_Run,k_Param}>(completedThreshold));
            meanCompleted{j_Task,i_Run,k_Param} = movmean(completed{j_Task,i_Run,k_Param},avgStep).*100;
        end
    end
end
%
figure(2)
clf
hold on
j_Task = 1;
nHypers = 11;
actualPlotNum = [1 2 3 4 5 6 7 8 9 10 11];
for i_Run=1:nRuns
    plotNum = mod(i_Run,nHypers);
    if plotNum == 0, plotNum = nHypers; end
    aplotNum = actualPlotNum(plotNum);
    subplot(3,4,aplotNum);
    hold on
    for k_Param = 1
        plot(meanCompleted{j_Task,i_Run,k_Param},'DisplayName',names(j_Task,i_Run,k_Param))
    end
    ylim([0 100])
    xlim([0 timestepLimit-timestepStart])
end
%legend
%
figure(1)
clf
hold on
for i_Run=1:nRuns
    plotNum = mod(i_Run,nHypers);
    if plotNum == 0, plotNum = nHypers; end
    aplotNum = actualPlotNum(plotNum);
    subplot(3,4,aplotNum);
    hold on
    for k_Param = 1
        plot(meanData{j_Task,i_Run,k_Param},'DisplayName',names(j_Task,i_Run,k_Param))
    end
    ylim([0 40])
    xlim([0 timestepLimit-timestepStart])
end
%legend

%%
nParams = 1;
for j_Task = 1:length(Tsk)
    for i_Run = 1:nParams
        for k_Param = 1:nRuns
            normData{j_Task,k_Param,i_Run} = Data{j_Task,k_Param,i_Run}/completedThreshold;
            normMeanData{j_Task,k_Param,i_Run} = movmean(normData{j_Task,k_Param,i_Run},avgStep);

            AvgData{j_Task,k_Param,i_Run} = mean(normData{j_Task,k_Param,i_Run});
            MaxData{j_Task,k_Param,i_Run} = max(normMeanData{j_Task,k_Param,i_Run});
            MaxCompeted{j_Task,k_Param,i_Run} = max(meanCompleted{j_Task,k_Param,i_Run});
            TotalCompleted{j_Task,k_Param,i_Run} = mean(Data{j_Task,k_Param,i_Run}>(completedThreshold)).*100;
        end
    end
end
%
AvgDataC = mean(reshape(AvgData,nParams,[]),2);
MaxDataC = mean(reshape(MaxData,nParams,[]),2);
MaxCompetedC = mean(reshape(MaxCompeted,nParams,[]),2);
TotalCompletedC = mean(reshape(TotalCompleted,nParams,[]),2);
AvgDataCstd = std(reshape(AvgData,nParams,[]),0,2);
MaxDataCstd = std(reshape(MaxData,nParams,[]),0,2);
MaxCompetedCstd = std(reshape(MaxCompeted,nParams,[]),0,2);
TotalCompletedCstd = std(reshape(TotalCompleted,nParams,[]),0,2);
%%
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
j_Task=1;
DataSlice = reshape(Data(j_Task,:,:),1,[]);
plotArray = cell2mat(DataSlice);
%plotArray = movmean(cell2mat(Data),100);
figure(7)
clf
grouping = [];
for i_Run=1:9
    grouping = [grouping repmat(i_Run,[1,10])]
end
boxplot(plotArray, grouping,'whisker',20)
