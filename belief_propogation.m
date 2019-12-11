function result = belief_propogation(size, array) 
    n = size; 
    m = n-1; 
    H=spalloc(n-1,n,2*(n-1)); 
    for i=1:n-1
         H(i, i)=1; 
        H(i, i+1)=1; 
    end
    G = generatormatrix(H, m , n); 
    spy(H);
    spy(G);
    H2DS(H, m, n); 
 
    k = n - m; 
    message = double(rand(k,1) >0.5); 
    x = G*message; 
    x_tilde = x;
    iscodeword(hard_decision(x_tilde, H), H);

    sigma = 0.5; 
    y = x_tilde+randn(n,1) * sigma^2; 
    TRIES = 100; t = 0; 
    while iscodeword(hard_decision(y, H), H) && t<TRIES
        y = x_tilde + randn(n,1) * sigma; 
        t = t+1; 
    end

    T=50; 
    u = 4*y/(2*sigma^2); 
    if ~any(mod(H*double(y<0),2))
        disp('y already represents a codeword');
        clf ; 
    else
        disp('performing BP');
        Y = iterate_BP(T,u); 
        for k=1:T
            if ~any(mod(H*double(Y(: ,k) <0),2))
                strcat('found a codeword at iteration _', num2str(k)) 
                break; 
            end
        end
        plot (0:T,Y,'o-'); 
    end

    result = iterate_BP(100, array');

    function bool = hard_decision(u, H)
         bool = ~any(mod(H*(double(u<0)),2));
    end

    function bool = iscodeword(y, H)
        bool = true;
    end

    function H = generate_H(m,n,d)
        H = sparse(m,n); 
        H = mod(H,2);
        while not( all (sum(H,1)>=2) && all (sum(H,2)>=2))
            H = H+abs(sprand(m,n,d))>0;
            H = mod(H,2);
        end
    end

    function [G] = generatormatrix(H, m , n)
        Hp = H; 
        colperm = 1:n;
        for j=1:m
            i=min(find(Hp( j :m, j ))); 
            if isempty(i)
                k=min(max(find(Hp(j ,:)) , j )); 
                if isempty(k)
                    disp (['problem in row', num2str(j ,0)]); 
                    continue; 
                end

                temp = Hp(: , j ); 
                Hp(: , j)=Hp(: ,k ); 
                Hp(: ,k)=temp; 
                temp=colperm(k ); 
                colperm(k)=colperm( j ); 
                colperm( j)=temp; 
            end

            i=i+j-1; 
            if (i~=j)
                temp = Hp(j ,:); 
                Hp(j ,:)=Hp(i ,:); 
                Hp(i ,:)=temp; 
            end

            K= find(Hp(: , j )); 
            K= K(find(K~=j)); 

            if(isempty(K))
             t1=full (Hp(j ,:)); 
             for k=K
                 t2=full (Hp(k ,:)); 
                 temp=xor(t1 , t2 ); 
                 Hp(k,:)=sparse(temp); 
             end
            end
        end

        A = Hp(: ,m+1:n);
        [b ,invperm] = sort(colperm); 
        G = [A; speye(n-m)]; 
        G = G(invperm, :); 
    end


% The belief propogation algorithm
 function [] = H2DS(H, m, n) 
     global B P S q;
     q = nnz(H);
     
      P = spalloc(q ,q ,(sum(H,2)-1)' * sum(H,2)); 
      S = spalloc(q ,q ,(sum(H,1)-1) * sum(H,1)');
      
      k=0; 
       for j=1:n 
           I=find(H(: , j )); 
           for x=1:length(I) 
               for y=x+1:length(I) 
                   P(k+x ,k+y)=1; 
                   P(k+y,k+x)=1; 
               end
           end
           k=k+length(I ); 
       end
       
        k=0; 
        for i=1:m
            J=find(H(i ,:)); 
            for x=1:length(J) 
                for y=x+1:length(J) 
                    S (k+x ,k+y)=1; 
                    S (k+y,k+x)=1; 
                end
            end
            k=k+length(J); 
        end
        
         B=spalloc(q ,n,q ); 
         b=[]; 
         for k=1:m
             b=[b find(H(k , : ) ) ] ; 
         end
         B=sparse ([1: q]', b', ones(q ,1) ,q ,n);     
 end
 
function y = S(x)
    global S_ q;
    
    y=ones(q ,1); 
    for i=1:q
        for j=find(S_(i ,:)) 
            y(i) = y(i) * tanh(x(j)/2); 
        end
    end
    y=2*atanh(y);
end

function y = iterate_BP(T,u) 
     global B P S q;
     
     x1_k = zeros(q ,1); 
     x2_k = zeros(q ,1); 
     x1_k_1 = zeros(q ,1); 
     x2_k_1 = zeros(q ,1);
     
      y=zeros(n,T+1); 
      for t=1:T
          x1_k_1 = P*x2_k + B*u; 
          x2_k_1 = S(threshold_1(x1_k)); 
          y(: , t) = B' * x2_k + u; 
          x1_k = x1_k_1; 
          x2_k = x2_k_1; 
      end 
end

function val = threshold_2(number)
    if number < 0
        number = 1;
    end
    val = int8(number) + 1;
    disp(val);
end

function val = threshold_1(number)
    val = int8(abs(number)) + 1;
end

function plot_BP_output(y, mono, filename)
    T=size(y); 
    T = T(2)-1;
     
     clf ; 
     if nargin>1
         plot (0:T,y,'ko-') % monochrome 
     else
         plot (0:T,y,'o-') % c o l o r 
     end
     grid on; 
     axis ([0 T min(min(y))-.5 max(max(y))+.5]) 
     LEGEND=[]; 
     for k=1:n
         LEGEND = [LEGEND ; strcat('output', num2str(k))]; 
     end
     legend(LEGEND) 
     xlabel('time'); 
     ylabel('LLR');
     
     if nargin==3
         if isstr (filename ) 
             saveas(gcf , filename ,'eps'); 
         end
     end
 end
 
 function [H, final_column_weights , final_row_weights] = MacKayNealCreateCode(n,r ,v ,h)
     m = floor(n*(1-r )); 
     H = zeros ([m,n]); 
     alpha = [];
     
      for i = 1:length(v) 
          for j = 1:( floor(v(i)*n)) 
              alpha = [ alpha i ]; 
          end
      end
      
      while (length(alpha) ~= n) 
          alpha = [ alpha i ]; 
      end
      beta = [];
      
      for i=1:length(h) 
          for j=1:(floor(h( i )*m)) 
              beta = [beta, i ]; 
          end
      end
      
      while(length(beta) ~= m) 
           beta = [beta i]; 
      end
      
       for i = 1:n 
           c = []; 
           beta_temp = beta; 
           for j = 1: alpha( i ) 
               temp_row = randi(1 ,1 ,[1 ,m]); 
                while ((( beta_temp(temp_row) == 0) && (max(beta_temp) > 0)) || (( beta_temp(temp_row) <= -1)))
                     temp row = mod(temp row+1,m)+1; 
                end
                c = [c temp row ]; 
                beta_temp(temp_row) = -10; 
           end

            for k = 1:length(c) 
                beta(c(k)) = beta(c(k))-1; 
            end

            for j = 1: alpha( i ) 
                H(c( j ) , i ) = 1; 
            end
       end
       
        column weights = H’?ones(m,1); 
        for i = 1:max(column_weights) 
            count = 0; 
            for j = 1:length(column_weights) 
                if (column_weights(j) == i) 
                    count = count + 1; 
                end
            end
            final_column_weights(i) = count/length(column_weights); 
        end
        
        row_weights = H*ones(n,1); 
        for i = 1:max(row_weights) 
            count = 0; 
            for j = 1:length(row_weights) 
                if (row_weights(j) == i) 
                    count = count + 1; 
                end
            end
            final_row_weights(i) = count/length(row_weights); 
        end
 end
 
end