%Train and test n 4x10x3 MLPs

%Initialize data and variables
x = csvread("..\Data\iris inputs.csv");
t = csvread("..\Data\iris classes.csv");

n = 500; %Number of Networks to Train
ep = 500; %Maximum Epochs

%Total Confusion
ctr_sum = 0;
cts_sum = 0;
ctr_mat_sum = 0;
cts_mat_sum = 0;

%Aggregate Outputs and Targets
y_Ttrain = zeros(3,n*120);
y_Ttest = zeros(3,n*30);
t_Ttrain = zeros(3,n*120);
t_Ttest = zeros(3,n*30);

%Total Performance
Ttr = 0;
Ttr_n = zeros(1,ep+1);

%Train Networks
for i = 1:n
    %Create Network
    net = newff(x, t, [12,8]); %Modfified for testing
    
    %Shuffle Data
    idx = randperm(length(x));
    x(:, idx) = x;
    t(:, idx) = t;
    
    %Divide training and testing sets
    x_train = x(:, 1:120);
    x_test = x(:, 121:150);
    t_train = t(:, 1:120); 
    t_test = t(:, 121:150);
    
    %Build and train network; hyperparameters modifited for testing
    net.trainParam.goal = 1e-10;
    net.trainParam.mu = 1e-2;
    net.layers{1}.transferFcn = "logsig";
    net.trainParam.epochs = ep;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net.divideParam.trainRatio = 1;
    [net, tr] = train(net, x_train, t_train);
    
    %Get Results
    
    %Performance
    if i==1
        Ttr = tr;
        Ttr.epoch = 0:ep;
    elseif length(Ttr.perf) >= length(tr.perf)
        Ttr.perf(1:length(tr.perf)) = Ttr.perf(1:length(tr.perf)) + tr.perf;
        Ttr.tperf(1:length(tr.tperf)) = Ttr.tperf(1:length(tr.tperf)) + tr.tperf;
        Ttr.vperf(1:length(tr.vperf)) = Ttr.vperf(1:length(tr.vperf)) + tr.vperf;
        Ttr.time(1:length(tr.time)) = Ttr.time(1:length(tr.time)) + tr.time;
        Ttr.gradient(1:length(tr.gradient)) = Ttr.gradient(1:length(tr.gradient)) + tr.gradient;
    else
        Ttr.perf = Ttr.perf + tr.perf(1:length(Ttr.perf));
        Ttr.perf(length(Ttr.perf)+1:length(tr.perf))= tr.perf(length(Ttr.perf)+1:length(tr.perf));
        Ttr.tperf = Ttr.tperf + tr.tperf(1:length(Ttr.tperf));
        Ttr.tperf(length(Ttr.tperf)+1:length(tr.tperf))= tr.tperf(length(Ttr.tperf)+1:length(tr.tperf));
        Ttr.vperf = Ttr.vperf + tr.vperf(1:length(Ttr.vperf));
        Ttr.vperf(length(Ttr.vperf)+1:length(tr.vperf))= tr.vperf(length(Ttr.vperf)+1:length(tr.vperf));
        Ttr.time = Ttr.time + tr.time(1:length(Ttr.time));
        Ttr.time(length(Ttr.time)+1:length(tr.time))= tr.time(length(Ttr.time)+1:length(tr.time));
        Ttr.gradient = Ttr.gradient + tr.gradient(1:length(Ttr.gradient));
        Ttr.gradient(length(Ttr.gradient)+1:length(tr.gradient))= tr.gradient(length(Ttr.gradient)+1:length(tr.gradient));
        
        Ttr.num_epochs = tr.num_epochs;
    end
    Ttr_n(1:length(tr.perf)) = Ttr_n(1:length(tr.perf))+ 1;
    
    %Sum Outputs and Targets
    y_train = net(x_train);
    y_test = net(x_test);
    t_Ttrain( : , ((i-1)*120)+1:i*120) = t_train;
    t_Ttest( : , ((i-1)*30)+1:i*30) = t_test;
    y_Ttrain( : , ((i-1)*120)+1:i*120) = y_train;
    y_Ttest( : , ((i-1)*30)+1:i*30) = y_test;

    %Sum Confusion
    [ctr, cmtr] = confusion(t_train, y_train);
    ctr_sum = ctr_sum + ctr;
    ctr_mat_sum = ctr_mat_sum + cmtr;
    [cts, cmts] = confusion(t_test, y_test);
    cts_sum = cts_sum + cts;
    cts_mat_sum = cts_mat_sum + cmts;
end

%Average Accuracy and Performance
avg_acts = 1-(cts_sum/n);
avg_actr = 1-(ctr_sum/n);
Ttr.perf = Ttr.perf./Ttr_n(1:length(Ttr.perf));
Ttr.vperf = Ttr.vperf./Ttr_n(1:length(Ttr.tperf));
Ttr.tperf = Ttr.tperf./Ttr_n(1:length(Ttr.vperf));
Ttr.time = Ttr.time./Ttr_n(1:length(Ttr.time));
Ttr.gradient = Ttr.gradient./Ttr_n(1:length(Ttr.gradient));
[best_perf, best_index] = min(Ttr.perf);
Ttr.best_epoch = best_index-1;
Ttr.best_perf = best_perf;

%Graphs
plotconfusion(t_Ttrain, y_Ttrain, "Training", t_Ttest, y_Ttest, "Testing");
figure;
plotperform(Ttr);

%Results
disp("Average training accuracy in " + num2str(n) + " trained networks: " + num2str(avg_actr));
disp(ctr_mat_sum');
disp("Average testing accuracy in " + num2str(n) + " trained networks: " + num2str(avg_acts));
disp(cts_mat_sum');
disp("Average network best performance: " + num2str(best_perf));
