%Train and test n 4x2 SLPs

%Initialize data and variables
x = csvread("..\Data\iris inputs.csv");
t = csvread("..\Data\iris classes.csv");

%Remove iris versicolour class and features
x = x(:, [1:50, 101:150]);
t = t([1,3], [1:50, 101:150]);

n = 500; %Number of Networks to Train
ep = 100; %Maximum Epochs

%Total Confusion
ctr_sum = 0;
cts_sum = 0;
ctr_mat_sum = 0;
cts_mat_sum = 0;

%Aggregate Outputs and Targets
y_Ttrain = zeros(2,n*80);
y_Ttest = zeros(2,n*20);
t_Ttrain = zeros(2,n*80);
t_Ttest = zeros(2,n*20);

%Total Performance
Ttr = 0;
Ttr_n = zeros(1,ep+1);

%Train Networks

for i = 1:n
    %Reset Network
    clear net;
    
    %Shuffle Data
    idx = randperm(length(x));
    x(:, idx) = x;
    t(:, idx) = t;

    %Divide training and testing sets
    x_train = x(:, 1:80);
    x_test = x(:, 81:100);
    t_train = t(:, 1:80); 
    t_test = t(:, 81:100);

    %Build and train network
    net = newp(x,t);
    net.trainParam.goal = 0;
    net.trainParam.epochs = ep;
    [net, tr] = train(net, x_train, t_train);
   
    %Get Results
    
    %Performance
    if i==1
        Ttr = tr;
    elseif length(Ttr.perf) >= length(tr.perf)
        Ttr.perf(1:length(tr.perf)) = Ttr.perf(1:length(tr.perf)) + tr.perf;     
    else
        Ttr.perf = Ttr.perf+ tr.perf(1:length(Ttr.perf));
        Ttr.perf(length(Ttr.perf)+1:length(tr.perf))= tr.perf(length(Ttr.perf)+1:length(tr.perf));
    end
    Ttr_n(1:length(tr.perf)) = Ttr_n(1:length(tr.perf))+ 1;
    
    
    %Sum Outputs and Targets
    y_train = net(x_train);
    y_test = net(x_test);
    t_Ttrain( : , ((i-1)*80)+1:i*80) = t_train;
    t_Ttest( : , ((i-1)*20)+1:i*20) = t_test;
    y_Ttrain( : , ((i-1)*80)+1:i*80) = y_train;
    y_Ttest( : , ((i-1)*20)+1:i*20) = y_test;
    
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
Ttr.perf = Ttr.perf./n;
avg_perf = mean(Ttr.perf);

%Graphs
plotconfusion(t_Ttrain, y_Ttrain, "Training", t_Ttest, y_Ttest, "Testing");
figure;
plotperform(Ttr);

%Results
disp("Average training accuracy in " + num2str(n) + " trained networks: " + num2str(avg_actr));
disp(ctr_mat_sum');
disp("Average testing accuracy in " + num2str(n) + " trained networks: " + num2str(avg_acts));
disp(cts_mat_sum');
disp("Average network best performance: " + num2str(avg_perf));
