function time=infertime(app)
x=linspace(0,length(app.data(:,1)),length(app.data(:,1)))';  
time= x .* app.si;
end