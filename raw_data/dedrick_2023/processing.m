
clear

global Sa Sna

function OD = GFPtoOD(g)

gn = g*1e-4;
OD = (gn<2.4).*(-0.0138*gn.^4 + 0.08916*gn.^3 - 0.2259*gn.^2 + 0.3505*gn + 0.005379) + (gn>=2.4).*min(0.33,20*(gn-2.4)+0.3201);

end
%% REMINDER: Use hours 3-8 to estimate c12 and c21 by fitting, rather than direct temporal estimation

% Using fit_logistics.m from https://www.mathworks.com/matlabcentral/fileexchange/41781-fit_logistic-t-q

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%Simulation Data
%Data derived from supernatant experiments performed with sGFP S. aureus,
%1850
%% Growth rates of monocultures in fresh media (enter from exp. data)
r1 = 1.85; %species 1 - Sa - from AA data 09JUL2020
r2 = 1.56; %species 2 - Sna1850 - from AA data 09JUL2020
%% Carrying capacities of monocultures in fresh media (enter from exp. data)
K1 = 0.61; %species 1 - Sa - from AA data 09JUL2020
K2 = 0.35; %species 2 - Sna1850 - from AA data 09JUL2020

%% Carrying capacities in supernatants (enter from exp. data)
K12 = 0.17; % species 1 in supernatant of species 2
K21 = 0.03; % species 2 in supernatant of species 1

%% Interaction coefficients
c12 = -1/K2*(K1-K12);
c21 = -1/K1*(K2-K21);

% initial ODs
S10 = 0.0005;
S20 = 0.005;

%% Simulate the dynamics
dt = 10/60; % time step, hrs
trng = dt*(1:96); %duration, time-steps
S1 = zeros(size(trng));
S2 = zeros(size(trng));

%% Assuming tl hour lag time at dilution step
t1l = 2.2; %2.35; %Sa - from AA data 09JUL2020
t2l = 1.8; %2.25; %Sna1850 - from AA data 09JUL2020

S1l(1) = S10;
S2l(1) = S20;

cnt = 1;
for t = trng(2:length(trng))
    cnt = cnt+1;
    if t>t1l
        S1l(cnt) = S1l(cnt-1) + dt*r1*S1l(cnt-1)*(1-(S1l(cnt-1)-c12*S2l(cnt-1))/K1);
    else
        S1l(cnt) = S1l(cnt-1);
   
    end
    
    if t>t2l
        S2l(cnt) = S2l(cnt-1) + dt*r2*S2l(cnt-1)*(1-(S2l(cnt-1)-c21*S1l(cnt-1))/K2);
    else
        S2l(cnt) = S2l(cnt-1);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Experimental Data

% select the data file
% [infile,inpath] = uigetfile('Z:\Members\Babak\Experiments\Analysis\Sandra\Coculture fluorescence\*.txt');
inpath = '';
infile = 'Cocultures_GFPSa_1850_1821_MC_Day1_29JUL2019.txt';
fid = fopen(strcat(inpath,infile),'r');

ch = '';

n = 0;
N = 96; %initially planned number of time points
Nr = 96; %max time actual reads from text file
dt = 10/60; % measurement time-step (hours)

r = 8; % number of rows
c = 12; % number of columns
Rd = zeros(1,Nr);
OD6 = zeros(r,c,Nr); % all OD values
GFP = zeros(r,c,Nr); % all GFP values
DsRed = zeros(r,c,Nr); % all DsRed values
ch = fscanf(fid,'%s',1);
while strcmp(ch,'Temperature')==0
    ch = fscanf(fid,'%s',1);
end
while strcmp(ch,'Time')==0
    ch = fscanf(fid,'%s',1);
end 
while n < Nr
    n = n+1;
    Rd(n) = fscanf(fid,'%i',1);
    Ts = fscanf(fid,'%s',1);
    Time(n,1:length(Ts)-2) = Ts(2:length(Ts)-1);
    ch = fscanf(fid,'%s',3);
    ch = fscanf(fid,'%s',13);
    for i = 1:r
        for j = 1:c
            OD6(i,j,n) = str2double(fscanf(fid,'%s',1));
        end
        ch = fscanf(fid,'%s',4);
    end
    ch = fscanf(fid,'%s',3);
end
while strcmp(ch,'Temperature:')==0
    ch = fscanf(fid,'%s',1);
end

while strcmp(ch,'Time')==0
    ch = fscanf(fid,'%s',1);
end
n = 0;
while n < Nr
    n = n+1;
    Rd(n) = fscanf(fid,'%i',1);
    Ts = fscanf(fid,'%s',1);
    TimeG(n,1:length(Ts)-2) = Ts(2:length(Ts)-1);
    ch = fscanf(fid,'%s',3);
    ch = fscanf(fid,'%s',13);
    for i = 1:r
        for j = 1:c
            GFP(i,j,n) = str2double(fscanf(fid,'%s',1));
        end
        ch = fscanf(fid,'%s',4);
    end
    ch = fscanf(fid,'%s',3);
end
for ndump = Nr+1:N
    ch = fscanf(fid,'%s',53);
end

ch=fclose(fid);

% Conversion of GFP to OD and DsRed to OD
%GFPtoOD = (GFP - mean(mean(GFP(1,1:8,1:10))))/192970;

% Build time vector
dt = 10/60;  % 10 minutes in hours
Nr = 96;
trng = dt*(1:Nr);  % Time in hours

repl = 2:7;  % Replicate rows (B–G)

cols = [5, 6];  % Columns 5 and 6
col_labels = {'1:1 Sa:Sna', '1:10 Sa:Sna'};

figure('Units','inches','Position',[1 1 10 4])  % Nice wide figure
for idx = 1:2
    col = cols(idx);

    % S. aureus OD - Experimental
    SaOD = GFPtoOD(shiftdim(GFP(repl, col, 1:Nr),2) - (mean(GFP(repl, col, 1:5),3)*ones(1,Nr))');
    % Total OD - Experimental
    TotalOD = shiftdim(OD6(repl, col, 1:Nr),2) - (mean(OD6(repl, col, 1:5),3)*ones(1,Nr))';
    % KPL1850 OD
    SnaOD = TotalOD - SaOD;

    % Average over replicates
    Sa_mean = nanmean(SaOD,2)';
    Sna_mean = nanmean(TotalOD,2)' - Sa_mean;
    
    Sa_mean(1) = 0;  % Force initial point to 0
    Sna_mean(1) = 0;

    % --- Plot ---
    subplot(1,2,idx)
    plot(trng, Sa_mean, 'b-', 'LineWidth', 2)
    hold on
    plot(trng, Sna_mean, 'r--', 'LineWidth', 2)
    xlabel('Time (hours)')
    ylabel('OD_{600}')
    title(col_labels{idx})
    legend('S. aureus (Sa)', 'KPL1850 (Sna)', 'Location', 'NorthWest')
    grid on
end

sgtitle('Growth Dynamics of S. aureus and KPL1850 (1:1 and 1:10 Conditions)')

% ---------------- Save to PDF ----------------
exportgraphics(gcf, 'GrowthDynamics_SideBySide.pdf', 'ContentType', 'vector')


%save
% ---------------- Save time and abundances to CSV ----------------

% For column 5 (1:1 Sa:Sna)
col = 5;

SaOD = GFPtoOD(shiftdim(GFP(repl, col, 1:Nr),2) - (mean(GFP(repl, col, 1:5),3)*ones(1,Nr))');
TotalOD = shiftdim(OD6(repl, col, 1:Nr),2) - (mean(OD6(repl, col, 1:5),3)*ones(1,Nr))';
SnaOD = TotalOD - SaOD;

Sa_mean = nanmean(SaOD,2)';
Sna_mean = nanmean(TotalOD,2)' - Sa_mean;

Sa_mean(1) = 0;
Sna_mean(1) = 0;

% Make table
T_1to1 = table(trng', Sa_mean', Sna_mean', 'VariableNames', {'Time_hr', 'Sa_OD', 'Sna_OD'});

% Save
writetable(T_1to1, 'growth_data_1to1.csv')

% ----------------

% For column 6 (1:10 Sa:Sna)
col = 6;

SaOD = GFPtoOD(shiftdim(GFP(repl, col, 1:Nr),2) - (mean(GFP(repl, col, 1:5),3)*ones(1,Nr))');
TotalOD = shiftdim(OD6(repl, col, 1:Nr),2) - (mean(OD6(repl, col, 1:5),3)*ones(1,Nr))';
SnaOD = TotalOD - SaOD;

Sa_mean = nanmean(SaOD,2)';
Sna_mean = nanmean(TotalOD,2)' - Sa_mean;

Sa_mean(1) = 0;
Sna_mean(1) = 0;

% Make table
T_1to10 = table(trng', Sa_mean', Sna_mean', 'VariableNames', {'Time_hr', 'Sa_OD', 'Sna_OD'});

% Save
writetable(T_1to10, 'growth_data_1to10.csv')
