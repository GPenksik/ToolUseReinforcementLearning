clear
n=20
p1 = (randn(n,1)).*5
p2 = (randn(n,1)).*2
x1 = [-0.45 -0.4 -0.35 -0.0]
x1 = [-0.8 -0.4 0.0 +0.4]
% x2 = [-0.2 -0.1 0 0.1]
%x = [-0.5 -0.4 -0.3 -0.2]
%%
for j=1:length(x1)
    for i=1:n
        y(j,i) = p1(i) * sin(p2(i)+x1(j)) 
        y(j,i) = y(j,i) + p1(n-i+1) * cos(p2(n-i+1)+x1(j)) 
%         y(j,i) = y(j,i) + p1(i) * sin(p2(i)*x2(j)) 
%         y(j,i) = y(j,i) + p1(n-i+1) * cos(p2(n-i+1)*x2(j)) 

    end
end
figure(1)
clf
hold on
for l=1:length(x1)
    plot(y(l,:))
end
legend