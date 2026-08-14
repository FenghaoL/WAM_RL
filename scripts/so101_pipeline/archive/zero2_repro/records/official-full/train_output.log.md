[launch] nproc_per_node=4 num_machines=1 machine_rank=0 task=robotwin_ahawam run_id=2026-08-11_14-09-08
W0811 14:09:14.099000 976400 site-packages/torch/distributed/run.py:766] 
W0811 14:09:14.099000 976400 site-packages/torch/distributed/run.py:766] *****************************************
W0811 14:09:14.099000 976400 site-packages/torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0811 14:09:14.099000 976400 site-packages/torch/distributed/run.py:766] *****************************************
08/11 [14:09:17] INFO     | >> Model init device resolved to cuda:0 (LOCAL_RANK=0,      runtime.py:554
                          current_cuda_device=0, visible_device_count=4)                              
08/11 [14:09:18] INFO     | >> Loading Wan2.2-TI2V-5B components...                      loader.py:152
08/11 [14:09:43] INFO     | >> Using gradient checkpointing for DiT blocks. This  wan_video_dit.py:412
                          will save memory but use more computation.                                  
08/11 [14:10:06] INFO     | >> Skipping pretrained text encoder/tokenizer load           loader.py:206
                          (`load_text_encoder=False`); training must provide cached                   
                          `context/context_mask`.                                                     
08/11 [14:10:12] INFO     | >> Finished loading Wan2.2-TI2V-5B components in 54.64       loader.py:211
                          seconds.                                                                    
08/11 [14:10:19] INFO     | >> ActionDiT backbone payload policy: {'skip_prefixes':  action_dit.py:186
                          ['action_encoder.', 'head.'], 'alpha_scaling': True,                        
                          'interpolation':                                                            
                          'sequential_1d_linear_align_corners_true'}                                  
08/11 [14:10:21] INFO     | >> Loaded ActionDiT backbone from                        action_dit.py:256
                          /home/fenghao/workspace/AHA-WAM/checkpoints/ActionDiT_line                  
                          ar_interp_Wan22_alphascale_1024hdim.pt (keys=820;                           
                          random_kept_prefixes=['action_encoder.', 'head.']).                         
08/11 [14:10:22] INFO     | >> Using gradient checkpointing for mixture attention. This     mot.py:165
                          will save memory but use more computation.                                  
                 INFO     | >> Initialized MoT with experts: ['video', 'action'],           mot.py:190
                          num_layers=30                                                               
                 INFO     | >>   Expert 'video': num_params=5.00 B                          mot.py:195
                 INFO     | >>   Expert 'action': num_params=1.02 B                         mot.py:195
08/11 [14:10:25] INFO     | >> PyTorch version 2.7.1+cu128 available.                     config.py:54
08/11 [14:10:29] INFO     | >> PyTorch version 2.7.1+cu128 available.                     config.py:54
08/11 [14:10:30] INFO     | >> PyTorch version 2.7.1+cu128 available.                     config.py:54
08/11 [14:10:33] INFO     | >> PyTorch version 2.7.1+cu128 available.                     config.py:54
08/11 [14:12:03] INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmp9_2auw7x/test.c -o /tmp/tmp9_2auw7x/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmp9_2auw7x/test.o -laio -o /tmp/tmp9_2auw7x/a.out               
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpp39pjlxl/test.c -o /tmp/tmpp39pjlxl/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpp39pjlxl/test.o                                               
                          -L/home/fenghao/miniconda3/envs/ahawam                                      
                          -L/home/fenghao/miniconda3/envs/ahawam/lib64 -lcufile -o                    
                          /tmp/tmpp39pjlxl/a.out                                                      
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpho0rr_vi/test.c -o /tmp/tmpho0rr_vi/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpho0rr_vi/test.o -laio -o /tmp/tmpho0rr_vi/a.out               
08/11 [14:12:03] INFO     | >> Using dataset stats:                         robot_video_dataset.py:201
                          checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.jso                           
                          n                                                                           
08/11 [14:12:04] INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpt6q_ixlw/test.c -o /tmp/tmpt6q_ixlw/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpt6q_ixlw/test.o -laio -o /tmp/tmpt6q_ixlw/a.out               
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmp8ali_jru/test.c -o /tmp/tmp8ali_jru/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmp8ali_jru/test.o                                               
                          -L/home/fenghao/miniconda3/envs/ahawam                                      
                          -L/home/fenghao/miniconda3/envs/ahawam/lib64 -lcufile -o                    
                          /tmp/tmp8ali_jru/a.out                                                      
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpge0nx1nj/test.c -o /tmp/tmpge0nx1nj/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpge0nx1nj/test.o -laio -o /tmp/tmpge0nx1nj/a.out               
08/11 [14:12:08] INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpea4__8li/test.c -o /tmp/tmpea4__8li/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpea4__8li/test.o -laio -o /tmp/tmpea4__8li/a.out               
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpm047fssb/test.c -o /tmp/tmpm047fssb/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpm047fssb/test.o                                               
                          -L/home/fenghao/miniconda3/envs/ahawam                                      
                          -L/home/fenghao/miniconda3/envs/ahawam/lib64 -lcufile -o                    
                          /tmp/tmpm047fssb/a.out                                                      
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpl330txkt/test.c -o /tmp/tmpl330txkt/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpl330txkt/test.o -laio -o /tmp/tmpl330txkt/a.out               
08/11 [14:12:17] INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpeehwc2c9/test.c -o /tmp/tmpeehwc2c9/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpeehwc2c9/test.o -laio -o /tmp/tmpeehwc2c9/a.out               
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpi6nkasc6/test.c -o /tmp/tmpi6nkasc6/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpi6nkasc6/test.o                                               
                          -L/home/fenghao/miniconda3/envs/ahawam                                      
                          -L/home/fenghao/miniconda3/envs/ahawam/lib64 -lcufile -o                    
                          /tmp/tmpi6nkasc6/a.out                                                      
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -march=x86-64-v2 -mtune=haswell -ftree-vectorize -fPIC                   
                          -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe             
                          -isystem /home/fenghao/miniconda3/envs/ahawam/include                       
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem                             
                          /home/fenghao/miniconda3/envs/ahawam/include                                
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -fPIC -march=x86-64-v2 -mtune=haswell -ftree-vectorize                
                          -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections             
                          -pipe -isystem /home/fenghao/miniconda3/envs/ahawam/include                 
                          -I/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/incl            
                          ude                                                                         
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs -c /tmp/tmpj2zupow0/test.c -o /tmp/tmpj2zupow0/test.o                 
                 INFO     | >>                                                             spawn.py:75
                          /home/fenghao/miniconda3/envs/ahawam/bin/x86_64-conda-linux-gnu-            
                          cc -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro                   
                          -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections                        
                          -Wl,-rpath,/home/fenghao/miniconda3/envs/ahawam/lib                         
                          -Wl,-rpath-link,/home/fenghao/miniconda3/envs/ahawam/lib                    
                          -L/home/fenghao/miniconda3/envs/ahawam/lib                                  
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib             
                          -L/home/fenghao/miniconda3/envs/ahawam/targets/x86_64-linux/lib/            
                          stubs /tmp/tmpj2zupow0/test.o -laio -o /tmp/tmpj2zupow0/a.out               
08/11 [14:12:17] INFO     | >> Accelerate training:                                      trainer.py:92
                          distributed_type=DistributedType.DEEPSPEED zero_stage=2                     
                          world_size=4 process_index=0 cfg_mixed_precision=bf16                       
                          accelerator_mixed_precision=bf16 grad_accum=1 grad_clip=1.0000              
                 INFO     | >> using accelerator.device=cuda:0                          trainer.py:103
08/11 [14:12:18] INFO     | >> Using history-aware train sampler.                       trainer.py:289
                 INFO     | >> LR scheduler: type=cosine class=CosineAnnealingLR        trainer.py:169
                          total_train_steps=1 warmup_ratio=0.05 warmup_steps=0                        
                          learning_rate=5e-05                                                         
Before initializing optimizer states
MA 21.85 GB         Max_MA 28.69 GB         CA 28.73 GB         Max_CA 29 GB 
CPU Virtual Memory:  used = 327.13 GB, percent = 65.0%
After initializing optimizer states
MA 21.85 GB         Max_MA 28.69 GB         CA 35.56 GB         Max_CA 36 GB 
CPU Virtual Memory:  used = 327.13 GB, percent = 65.0%
After initializing ZeRO optimizer
MA 21.85 GB         Max_MA 21.85 GB         CA 35.56 GB         Max_CA 36 GB 
CPU Virtual Memory:  used = 327.14 GB, percent = 65.0%
08/11 [14:12:21] INFO     | >> Train/val dataset size: 6011575/6011575                  trainer.py:210
                 INFO     | >> Setting DiT to train mode and freezing other model       trainer.py:682
                          components.                                                                 
                 INFO     | >> Starting training with max_steps=1.                     trainer.py:1061
                 INFO     | >> Disabled automatic GC to avoid bwd stalls with gradient trainer.py:1067
                          checkpointing.                                                              
Error executing job with overrides: ['wandb.name=robotwin_ahawam', 'task=robotwin_ahawam', 'model=ahawam', 'model.action_dit_pretrained_path=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt', 'data.train.dataset_dirs=[data/robotwin2.0-fastwam/robotwin2.0]', 'data.train.pretrained_norm_stats=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json', 'data.train.text_embedding_cache_dir=data/robotwin_text_embeds_cache', 'data.val=null', 'output_dir=runs/zero2_repro/official-full/model', 'wandb.enabled=false', 'batch_size=1', 'num_workers=0', 'gradient_accumulation_steps=1', 'max_steps=1', 'log_every=1', 'save_every=1000000', 'eval_every=0', 'freeze_video_dit=false', 'model.mot_checkpoint_mixed_attn=true']
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 38, in <module>
[rank1]:     main()
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
[rank1]:     _run_hydra(
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
[rank1]:     _run_app(
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
[rank1]:     run_and_report(
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
[rank1]:     raise ex
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
[rank1]:     return func()
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
[rank1]:     lambda: hydra.run(
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
[rank1]:     _ = ret.return_value
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
[rank1]:     raise self._return_value
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
[rank1]:     ret.return_value = task_function(task_cfg)
[rank1]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 34, in main
[rank1]:     run_training(cfg)
[rank1]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/runtime.py", line 570, in run_training
[rank1]:     trainer.train()
[rank1]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/trainer.py", line 1100, in train
[rank1]:     self.accelerator.backward(loss)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/accelerator.py", line 2844, in backward
[rank1]:     self.deepspeed_engine_wrapped.backward(loss, sync_gradients=self.sync_gradients, **kwargs)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/utils/deepspeed.py", line 281, in backward
[rank1]:     self.engine.step()
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2771, in step
[rank1]:     self._take_model_step(lr_kwargs)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2666, in _take_model_step
[rank1]:     self.optimizer.step()
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2178, in step
[rank1]:     self._optimizer_step(i)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2081, in _optimizer_step
[rank1]:     self.optimizer.step()
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/lr_scheduler.py", line 124, in wrapper
[rank1]:     return func.__get__(opt, opt.__class__)(*args, **kwargs)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 485, in wrapper
[rank1]:     out = func(*args, **kwargs)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 79, in _use_grad
[rank1]:     ret = func(self, *args, **kwargs)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 236, in step
[rank1]:     has_complex = self._init_group(
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 838, in _fn
[rank1]:     return fn(*args, **kwargs)
[rank1]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 180, in _init_group
[rank1]:     state["exp_avg_sq"] = torch.zeros_like(
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.85 GiB. GPU 1 has a total capacity of 39.49 GiB of which 2.26 GiB is free. Including non-PyTorch memory, this process has 37.21 GiB memory in use. Of the allocated memory 35.57 GiB is allocated by PyTorch, and 62.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Error executing job with overrides: ['wandb.name=robotwin_ahawam', 'task=robotwin_ahawam', 'model=ahawam', 'model.action_dit_pretrained_path=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt', 'data.train.dataset_dirs=[data/robotwin2.0-fastwam/robotwin2.0]', 'data.train.pretrained_norm_stats=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json', 'data.train.text_embedding_cache_dir=data/robotwin_text_embeds_cache', 'data.val=null', 'output_dir=runs/zero2_repro/official-full/model', 'wandb.enabled=false', 'batch_size=1', 'num_workers=0', 'gradient_accumulation_steps=1', 'max_steps=1', 'log_every=1', 'save_every=1000000', 'eval_every=0', 'freeze_video_dit=false', 'model.mot_checkpoint_mixed_attn=true']
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 38, in <module>
[rank0]:     main()
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
[rank0]:     _run_hydra(
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
[rank0]:     _run_app(
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
[rank0]:     run_and_report(
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
[rank0]:     raise ex
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
[rank0]:     return func()
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
[rank0]:     lambda: hydra.run(
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
[rank0]:     _ = ret.return_value
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
[rank0]:     raise self._return_value
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
[rank0]:     ret.return_value = task_function(task_cfg)
[rank0]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 34, in main
[rank0]:     run_training(cfg)
[rank0]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/runtime.py", line 570, in run_training
[rank0]:     trainer.train()
[rank0]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/trainer.py", line 1100, in train
[rank0]:     self.accelerator.backward(loss)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/accelerator.py", line 2844, in backward
[rank0]:     self.deepspeed_engine_wrapped.backward(loss, sync_gradients=self.sync_gradients, **kwargs)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/utils/deepspeed.py", line 281, in backward
[rank0]:     self.engine.step()
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2771, in step
[rank0]:     self._take_model_step(lr_kwargs)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2666, in _take_model_step
[rank0]:     self.optimizer.step()
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2178, in step
[rank0]:     self._optimizer_step(i)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2081, in _optimizer_step
[rank0]:     self.optimizer.step()
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/lr_scheduler.py", line 124, in wrapper
[rank0]:     return func.__get__(opt, opt.__class__)(*args, **kwargs)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 485, in wrapper
[rank0]:     out = func(*args, **kwargs)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 79, in _use_grad
[rank0]:     ret = func(self, *args, **kwargs)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 236, in step
[rank0]:     has_complex = self._init_group(
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 838, in _fn
[rank0]:     return fn(*args, **kwargs)
[rank0]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 180, in _init_group
[rank0]:     state["exp_avg_sq"] = torch.zeros_like(
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.85 GiB. GPU 0 has a total capacity of 39.49 GiB of which 2.25 GiB is free. Including non-PyTorch memory, this process has 37.21 GiB memory in use. Of the allocated memory 35.57 GiB is allocated by PyTorch, and 64.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Error executing job with overrides: ['wandb.name=robotwin_ahawam', 'task=robotwin_ahawam', 'model=ahawam', 'model.action_dit_pretrained_path=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt', 'data.train.dataset_dirs=[data/robotwin2.0-fastwam/robotwin2.0]', 'data.train.pretrained_norm_stats=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json', 'data.train.text_embedding_cache_dir=data/robotwin_text_embeds_cache', 'data.val=null', 'output_dir=runs/zero2_repro/official-full/model', 'wandb.enabled=false', 'batch_size=1', 'num_workers=0', 'gradient_accumulation_steps=1', 'max_steps=1', 'log_every=1', 'save_every=1000000', 'eval_every=0', 'freeze_video_dit=false', 'model.mot_checkpoint_mixed_attn=true']
[rank3]: Traceback (most recent call last):
[rank3]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 38, in <module>
[rank3]:     main()
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
[rank3]:     _run_hydra(
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
[rank3]:     _run_app(
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
[rank3]:     run_and_report(
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
[rank3]:     raise ex
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
[rank3]:     return func()
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
[rank3]:     lambda: hydra.run(
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
[rank3]:     _ = ret.return_value
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
[rank3]:     raise self._return_value
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
[rank3]:     ret.return_value = task_function(task_cfg)
[rank3]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 34, in main
[rank3]:     run_training(cfg)
[rank3]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/runtime.py", line 570, in run_training
[rank3]:     trainer.train()
[rank3]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/trainer.py", line 1100, in train
[rank3]:     self.accelerator.backward(loss)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/accelerator.py", line 2844, in backward
[rank3]:     self.deepspeed_engine_wrapped.backward(loss, sync_gradients=self.sync_gradients, **kwargs)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/utils/deepspeed.py", line 281, in backward
[rank3]:     self.engine.step()
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2771, in step
[rank3]:     self._take_model_step(lr_kwargs)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2666, in _take_model_step
[rank3]:     self.optimizer.step()
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2178, in step
[rank3]:     self._optimizer_step(i)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2081, in _optimizer_step
[rank3]:     self.optimizer.step()
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/lr_scheduler.py", line 124, in wrapper
[rank3]:     return func.__get__(opt, opt.__class__)(*args, **kwargs)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 485, in wrapper
[rank3]:     out = func(*args, **kwargs)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 79, in _use_grad
[rank3]:     ret = func(self, *args, **kwargs)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 236, in step
[rank3]:     has_complex = self._init_group(
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 838, in _fn
[rank3]:     return fn(*args, **kwargs)
[rank3]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 180, in _init_group
[rank3]:     state["exp_avg_sq"] = torch.zeros_like(
[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.85 GiB. GPU 3 has a total capacity of 39.49 GiB of which 2.25 GiB is free. Including non-PyTorch memory, this process has 37.21 GiB memory in use. Of the allocated memory 35.57 GiB is allocated by PyTorch, and 64.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Error executing job with overrides: ['wandb.name=robotwin_ahawam', 'task=robotwin_ahawam', 'model=ahawam', 'model.action_dit_pretrained_path=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt', 'data.train.dataset_dirs=[data/robotwin2.0-fastwam/robotwin2.0]', 'data.train.pretrained_norm_stats=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json', 'data.train.text_embedding_cache_dir=data/robotwin_text_embeds_cache', 'data.val=null', 'output_dir=runs/zero2_repro/official-full/model', 'wandb.enabled=false', 'batch_size=1', 'num_workers=0', 'gradient_accumulation_steps=1', 'max_steps=1', 'log_every=1', 'save_every=1000000', 'eval_every=0', 'freeze_video_dit=false', 'model.mot_checkpoint_mixed_attn=true']
[rank2]: Traceback (most recent call last):
[rank2]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 38, in <module>
[rank2]:     main()
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
[rank2]:     _run_hydra(
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
[rank2]:     _run_app(
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
[rank2]:     run_and_report(
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
[rank2]:     raise ex
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
[rank2]:     return func()
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
[rank2]:     lambda: hydra.run(
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
[rank2]:     _ = ret.return_value
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
[rank2]:     raise self._return_value
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
[rank2]:     ret.return_value = task_function(task_cfg)
[rank2]:   File "/home/fenghao/workspace/AHA-WAM/scripts/train.py", line 34, in main
[rank2]:     run_training(cfg)
[rank2]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/runtime.py", line 570, in run_training
[rank2]:     trainer.train()
[rank2]:   File "/home/fenghao/workspace/AHA-WAM/src/ahawam/trainer.py", line 1100, in train
[rank2]:     self.accelerator.backward(loss)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/accelerator.py", line 2844, in backward
[rank2]:     self.deepspeed_engine_wrapped.backward(loss, sync_gradients=self.sync_gradients, **kwargs)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/utils/deepspeed.py", line 281, in backward
[rank2]:     self.engine.step()
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2771, in step
[rank2]:     self._take_model_step(lr_kwargs)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 2666, in _take_model_step
[rank2]:     self.optimizer.step()
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2178, in step
[rank2]:     self._optimizer_step(i)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 2081, in _optimizer_step
[rank2]:     self.optimizer.step()
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/lr_scheduler.py", line 124, in wrapper
[rank2]:     return func.__get__(opt, opt.__class__)(*args, **kwargs)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 485, in wrapper
[rank2]:     out = func(*args, **kwargs)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/optimizer.py", line 79, in _use_grad
[rank2]:     ret = func(self, *args, **kwargs)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 236, in step
[rank2]:     has_complex = self._init_group(
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 838, in _fn
[rank2]:     return fn(*args, **kwargs)
[rank2]:   File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/optim/adam.py", line 180, in _init_group
[rank2]:     state["exp_avg_sq"] = torch.zeros_like(
[rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.85 GiB. GPU 2 has a total capacity of 39.49 GiB of which 2.25 GiB is free. Including non-PyTorch memory, this process has 37.22 GiB memory in use. Of the allocated memory 35.57 GiB is allocated by PyTorch, and 66.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank0]:[W811 14:12:34.373071762 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0811 14:12:36.277000 976400 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 976887 closing signal SIGTERM
W0811 14:12:36.277000 976400 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 976888 closing signal SIGTERM
W0811 14:12:36.278000 976400 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 976889 closing signal SIGTERM
E0811 14:12:37.207000 976400 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 0 (pid: 976886) of binary: /home/fenghao/miniconda3/envs/ahawam/bin/python3.10
Traceback (most recent call last):
  File "/home/fenghao/miniconda3/envs/ahawam/bin/accelerate", line 6, in <module>
    sys.exit(main())
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 50, in main
    args.func(args)
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1266, in launch_command
    deepspeed_launcher(args)
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/commands/launch.py", line 952, in deepspeed_launcher
    distrib_run.run(args)
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/distributed/run.py", line 883, in run
    elastic_launch(
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 270, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
scripts/train.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-08-11_14:12:36
  host      : chickadee.engin.umich.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 976886)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
