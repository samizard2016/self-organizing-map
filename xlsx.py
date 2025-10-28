from think_excel_1 import ThinkExcel
class XLSX:
    def __init__(self,xlsx_file):
        self.output_file = xlsx_file
        self.set_output_excel()
    def set_output_excel(self):
        self.xl = ThinkExcel()
        self.wb = self.xl.create_wb(self.output_file)
    def write_to_ws(self,df,sheet,dispTabName):
        self.xl.add_df_to_ws(sheet,new=True,df=df)
        ref = self.xl.get_ref(start_row=0,start_col=0,end_row=df.shape[0]+1,end_col=df.shape[1])
        self.xl.add_table_to_ws(dispTabName,ref)
    def add_header_footer(self,header,footer):
        self.xl.set_header_to_ws(header)
        self.xl.set_footer_to_ws(footer)
    def save_excel(self):
        self.xl.save_current_wb()